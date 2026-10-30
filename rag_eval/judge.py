"""LLM-as-Judge：用大模型按统一量规给答案打分，并与原生指标交叉校验。

两种模式：
- ``llm-judge``：有可用 LLM 时，把「问题 + 检索片段 + 答案 + 参考答案」交给模型，
  要求输出结构化 JSON（忠实度/相关性/完整性/引用质量 + 理由）。这是 RAGAs、
  DeepEval 之外一道独立的评判视角，能发现词法指标漏掉的语义问题。
- ``heuristic-judge``：无 LLM 时的确定性量规评审，逐句检查可支撑性并生成理由，
  保证离线环境仍能演示「评判 + 解释」的完整流程。

评审结果会与原生指标一起展示，报告里给出两者的差值，用于发现「指标虚高」。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import EvaluateCfg
from .generate import format_contexts
from .llm import ChatMessage, LLMClient, LLMError
from .metrics.base import keyword_score, sentence_support
from .text import (
    coverage,
    extract_citations,
    is_refusal,
    sentences,
    strip_citations,
    term_set,
)
from .types import CaseSpec, Retrieved

JUDGE_SYSTEM = (
    "你是严格的 RAG 系统评测员。你会拿到用户问题、检索片段、系统答案和参考答案，"
    "请只依据给出的材料打分，不要使用外部知识。"
)

JUDGE_TEMPLATE = """请对下列答案打分，每个维度 0~1，输出 JSON（不要输出其它内容）：
{{"faithfulness": 0~1, "relevance": 0~1, "completeness": 0~1, "citation_quality": 0~1, "reason": "不超过 80 字的中文理由"}}

评分标准：
- faithfulness：答案中每个事实是否都能在检索片段中找到依据；出现片段未提及的数字或结论即扣分。
- relevance：答案是否直接回应了用户问题，而不是罗列背景。
- completeness：参考答案中的关键信息点是否都被覆盖。
- citation_quality：引用编号是否指向真正支撑该句的片段，是否缺引用。

参考片段：
{context}

用户问题：{question}

参考答案（可能为空）：{reference}

系统答案：
{answer}
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeOutput:
    scores: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    source: str = "heuristic-judge"
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "reasons": self.reasons,
            "source": self.source,
        }


class Judge:
    def __init__(self, cfg: EvaluateCfg, llm: LLMClient | None = None):
        self.cfg = cfg
        self.llm = llm

    @property
    def available(self) -> bool:
        return self.llm is not None

    @property
    def source(self) -> str:
        return "llm-judge" if self.available else "heuristic-judge"

    def evaluate(
        self,
        case: CaseSpec,
        answer: str,
        contexts: Sequence[Retrieved],
        idf: dict[str, float],
        refused: bool = False,
    ) -> JudgeOutput:
        is_ref = refused or is_refusal(answer)
        if self.llm is not None:
            try:
                return self._llm_judge(case, answer, contexts)
            except (LLMError, ValueError, KeyError) as exc:
                output = self._heuristic_judge(case, answer, contexts, idf, is_ref)
                output.reasons["error"] = f"LLM 评审失败，已回退启发式：{exc}"
                return output
        return self._heuristic_judge(case, answer, contexts, idf, is_ref)

    # -- LLM 评审 --------------------------------------------------------
    def _llm_judge(self, case: CaseSpec, answer: str, contexts: Sequence[Retrieved]) -> JudgeOutput:
        prompt = JUDGE_TEMPLATE.format(
            context=format_contexts(contexts, 3000),
            question=case.question,
            reference=case.reference_answer or "（无）",
            answer=answer,
        )
        response = self.llm.complete(  # type: ignore[union-attr]
            [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", prompt)],
            temperature=self.cfg.judge_temperature,
            max_tokens=400,
            json_mode=True,
        )
        text = response.text.strip()
        match = _JSON_RE.search(text)
        if not match:
            raise ValueError(f"评审输出不是 JSON：{text[:120]}")
        payload = json.loads(match.group(0))
        scores: dict[str, float] = {}
        for key in ("faithfulness", "relevance", "completeness", "citation_quality"):
            if key in payload:
                scores[key] = max(0.0, min(1.0, float(payload[key])))
        if not scores:
            raise ValueError("评审输出缺少有效维度")
        return JudgeOutput(
            scores=scores,
            reasons={"overall": str(payload.get("reason") or "")[:300]},
            source="llm-judge",
            raw=text[:1000],
        )

    # -- 启发式评审 ------------------------------------------------------
    def _heuristic_judge(
        self,
        case: CaseSpec,
        answer: str,
        contexts: Sequence[Retrieved],
        idf: dict[str, float],
        refused: bool,
    ) -> JudgeOutput:
        if refused:
            appropriate = 1.0 if case.should_refuse else 0.0
            return JudgeOutput(
                scores={
                    "faithfulness": 1.0,
                    "relevance": appropriate,
                    "completeness": appropriate,
                    "citation_quality": 1.0 if appropriate else 0.0,
                },
                reasons={"overall": "识别为拒答；该题本应拒答。" if appropriate else "识别为拒答，但该题本应正常作答。"},
                source="heuristic-judge",
            )

        sents = [strip_citations(s) for s in sentences(answer) if len(term_set(strip_citations(s))) >= 2]
        supports = [sentence_support(s, contexts, idf) for s in sents]
        weak = [(s, sup) for s, sup in zip(sents, supports) if sup < 0.6]
        faithful = 1.0 - (len(weak) / len(sents)) if sents else 1.0

        q_terms = term_set(case.question)
        a_terms = term_set(strip_citations(answer))
        relevance = coverage(q_terms, a_terms, idf)
        completeness = keyword_score(case, strip_citations(answer))

        cited = sum(1 for s in sentences(answer) if extract_citations(s))
        citation_quality = min(1.0, cited / len(sents)) if sents else 0.0

        reason_parts: list[str] = []
        if weak:
            sentence, support = max(weak, key=lambda item: len(item[0]))
            reason_parts.append(f"「{sentence[:40]}」缺乏片段支撑（支撑度 {support:.2f}）")
        if completeness < 0.6:
            reason_parts.append("参考答案中的部分要点未覆盖")
        if citation_quality < 1.0:
            reason_parts.append("存在未标注引用的断言")
        reason = "；".join(reason_parts) if reason_parts else "各维度均无明显问题"
        return JudgeOutput(
            scores={
                "faithfulness": max(0.0, min(1.0, faithful)),
                "relevance": max(0.0, min(1.0, relevance)),
                "completeness": max(0.0, min(1.0, completeness)),
                "citation_quality": max(0.0, min(1.0, citation_quality)),
            },
            reasons={"overall": reason[:300]},
            source="heuristic-judge",
        )

    def info(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "available": self.available,
            "model": getattr(self.llm, "model", "heuristic-rubric-v1") if self.available else "heuristic-rubric-v1",
        }
