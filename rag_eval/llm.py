"""LLM 接入层：OpenAI 兼容 Chat Completions 客户端（标准库 urllib 实现）。

支持 DeepSeek / Qwen / Moonshot / vLLM / Ollama 等所有 OpenAI 兼容端点；
没有 API Key 时不报错，由上层降级到离线生成器，保证评测流水线可离线运行。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import GenerateCfg


class LLMError(RuntimeError):
    """LLM 调用失败（网络、鉴权、限流或返回体异常）。"""


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 2),
        }


class LLMClient(Protocol):
    name: str
    kind: str

    def complete(self, messages: Sequence[ChatMessage], *, temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False) -> LLMResponse: ...

    def info(self) -> dict[str, Any]: ...


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 约 1.6 字/token，拉丁约 4 字符/token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.6 + other / 4.0))


class OpenAICompatClient:
    """任意 OpenAI 兼容端点的最小客户端，带重试与成本核算。"""

    kind = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        cost_input_per_1k: float = 0.0,
        cost_output_per_1k: float = 0.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cost_input_per_1k = cost_input_per_1k
        self.cost_output_per_1k = cost_output_per_1k
        self.extra_headers = extra_headers or {}
        self.name = f"openai-compat:{model}"
        self.calls = 0
        self.errors = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "kb-rag-eval/1.0",
        }
        headers.update(self.extra_headers)
        return headers

    def _payload(self, messages: Sequence[ChatMessage], temperature: float, max_tokens: int, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError("缺少 API Key：请设置 RAG_EVAL_API_KEY 或 OPENAI_API_KEY")
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(
            self._payload(
                messages,
                self.temperature if temperature is None else temperature,
                self.max_tokens if max_tokens is None else max_tokens,
                json_mode,
            ),
            ensure_ascii=False,
        ).encode("utf-8")
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                latency = (time.perf_counter() - started) * 1000
                text = ""
                try:
                    text = raw["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    raise LLMError(f"响应体缺少 choices[0].message.content：{str(raw)[:300]}")
                usage = raw.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens("".join(m.content for m in messages)))
                completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
                cost = (
                    prompt_tokens / 1000.0 * self.cost_input_per_1k
                    + completion_tokens / 1000.0 * self.cost_output_per_1k
                )
                self.calls += 1
                return LLMResponse(
                    text=text.strip(),
                    model=str(raw.get("model") or self.model),
                    provider=self.name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost,
                    latency_ms=latency,
                    raw={"id": raw.get("id"), "finish_reason": (raw.get("choices") or [{}])[0].get("finish_reason")},
                )
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    detail = ""
                last_error = LLMError(f"HTTP {exc.code}：{detail or exc.reason}")
                if exc.code in (401, 403, 400, 404):
                    break  # 鉴权/参数错误，重试无意义
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"请求失败：{exc}")
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
            self.errors += 1
        raise last_error or LLMError("未知 LLM 调用错误")

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.kind,
            "name": self.name,
            "model": self.model,
            "base_url": self.base_url,
            "calls": self.calls,
            "errors": self.errors,
            "mode": "live",
        }


class EchoLLM:
    """测试用：原样回显最后一条用户消息或返回固定脚本。"""

    kind = "echo"

    def __init__(self, script: Sequence[str] | None = None):
        self.name = "echo"
        self.script = list(script or [])
        self.calls = 0

    def complete(self, messages, *, temperature: float = 0.0, max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        self.calls += 1
        if self.script:
            text = self.script.pop(0)
        else:
            text = messages[-1].content if messages else ""
        return LLMResponse(text=text, model="echo", provider="echo")

    def info(self) -> dict[str, Any]:
        return {"provider": "echo", "name": self.name, "mode": "test"}


def has_api_key(cfg: GenerateCfg) -> bool:
    return bool(cfg.api_key_env and os.environ.get(cfg.api_key_env)) or bool(os.environ.get("OPENAI_API_KEY"))


def build_llm(cfg: GenerateCfg, api_key: str | None = None) -> LLMClient | None:
    """按配置构造 LLM 客户端；``offline`` 或缺少密钥时返回 None（上层降级）。"""
    provider = (cfg.provider or "auto").lower()
    if provider == "offline":
        return None
    key = api_key if api_key is not None else (
        os.environ.get(cfg.api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
    )
    if not key:
        return None
    return OpenAICompatClient(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        cost_input_per_1k=cfg.cost_input_per_1k,
        cost_output_per_1k=cfg.cost_output_per_1k,
    )
