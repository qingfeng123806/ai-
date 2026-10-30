"""第三方评测框架适配器：RAGAs 与 DeepEval（可选依赖，缺失时自动降级）。

设计原则：适配器只做「翻译」——把本系统的 CaseResult 翻译成框架需要的样本结构，
再把框架的分数按 ``case_id`` 映射回本系统。框架缺失、模型不可用或调用失败时，
适配器返回空结果并把原因写进状态，主流程继续用原生指标，评测不会中断。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Config
from .types import CaseResult, CaseSpec


class AdapterUnavailable(RuntimeError):
    pass


@dataclass
class AdapterStatus:
    name: str
    available: bool
    metrics: list[str] = field(default_factory=list)
    reason: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "metrics": self.metrics,
            "reason": self.reason,
            "version": self.version,
        }


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "") or "")


def ragas_status(cfg: Config) -> AdapterStatus:
    if not cfg.evaluate.ragas:
        return AdapterStatus("ragas", False, reason="配置中已关闭 evaluate.ragas")
    try:
        import ragas  # type: ignore
    except Exception as exc:
        return AdapterStatus("ragas", False, reason=f"未安装 ragas：{exc}")
    if not (os.environ.get(cfg.generate.api_key_env) or os.environ.get("OPENAI_API_KEY")):
        return AdapterStatus("ragas", False, reason="未配置 LLM API Key，RAGAs 需要调用模型打分", version=_module_version(ragas))
    return AdapterStatus(
        "ragas",
        True,
        metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        version=_module_version(ragas),
    )


def deepeval_status(cfg: Config) -> AdapterStatus:
    if not cfg.evaluate.deepeval:
        return AdapterStatus("deepeval", False, reason="配置中已关闭 evaluate.deepeval")
    try:
        import deepeval  # type: ignore
    except Exception as exc:
        return AdapterStatus("deepeval", False, reason=f"未安装 deepeval：{exc}")
    if not (os.environ.get(cfg.generate.api_key_env) or os.environ.get("OPENAI_API_KEY")):
        return AdapterStatus("deepeval", False, reason="未配置 LLM API Key，DeepEval 需要调用模型打分", version=_module_version(deepeval))
    return AdapterStatus(
        "deepeval",
        True,
        metrics=["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"],
        version=_module_version(deepeval),
    )


# --------------------------------------------------------------------------
# RAGAs
# --------------------------------------------------------------------------

class RagasAdapter:
    """把本系统样本交给 RAGAs 重新打分，返回 {case_id: {metric: score}}。"""

    name = "ragas"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.status = ragas_status(cfg)

    @property
    def available(self) -> bool:
        return self.status.available

    def _llm(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as exc:
            raise AdapterUnavailable(
                "RAGAs 需要 langchain-openai：pip install langchain-openai"
            ) from exc
        key = os.environ.get(self.cfg.generate.api_key_env) or os.environ.get("OPENAI_API_KEY")
        return ChatOpenAI(
            model=self.cfg.generate.model,
            api_key=key,  # type: ignore[arg-type]
            base_url=self.cfg.generate.base_url,
            temperature=0.0,
        )

    def evaluate(self, cases: Sequence[CaseSpec], results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
        if not self.available:
            return {}
        try:
            from ragas import evaluate as ragas_evaluate  # type: ignore
            from ragas.metrics import context_precision, context_recall, faithfulness  # type: ignore
        except Exception as exc:
            self.status.reason = f"RAGAs 导入失败：{exc}"
            self.status.available = False
            return {}
        try:
            from datasets import Dataset  # type: ignore
        except Exception as exc:
            self.status.reason = f"RAGAs 需要 datasets：{exc}"
            self.status.available = False
            return {}

        payload = {
            "question": [c.question for c in cases],
            "answer": [r.answer for r in results],
            "contexts": [[item.chunk.text for item in r.contexts] for r in results],
            "ground_truth": [c.reference_answer or " " for c in cases],
        }
        try:
            dataset = Dataset.from_dict(payload)
            scores = ragas_evaluate(
                dataset,
                metrics=[faithfulness, context_precision, context_recall],
                llm=self._llm(),
                raise_exceptions=False,
            )
            frame = scores.to_pandas()
        except Exception as exc:  # pragma: no cover - 需真实依赖与网络
            self.status.reason = f"RAGAs 评测失败：{exc}"
            return {}

        mapping: dict[str, dict[str, float]] = {}
        for index, case in enumerate(cases):
            row = frame.iloc[index].to_dict()
            row_scores: dict[str, float] = {}
            for key in ("faithfulness", "context_precision", "context_recall"):
                if key in row and row[key] is not None:
                    try:
                        row_scores[key] = float(row[key])
                    except (TypeError, ValueError):
                        continue
            if row_scores:
                mapping[case.case_id] = row_scores
        return mapping


# --------------------------------------------------------------------------
# DeepEval
# --------------------------------------------------------------------------

class DeepEvalAdapter:
    """使用 DeepEval 的 G-Eval 系列指标复算忠实度与上下文精确率/召回率。"""

    name = "deepeval"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.status = deepeval_status(cfg)

    @property
    def available(self) -> bool:
        return self.status.available

    def _model(self) -> Any:
        try:
            from deepeval.models import DeepEvalBaseLLM  # type: ignore
        except Exception as exc:
            raise AdapterUnavailable("DeepEval 导入失败") from exc
        from .llm import ChatMessage, build_llm

        client = build_llm(self.cfg.generate)
        if client is None:
            raise AdapterUnavailable("DeepEval 需要可用的 LLM 客户端")

        class _Wrapper(DeepEvalBaseLLM):  # pragma: no cover - 需真实依赖
            def __init__(self) -> None:
                self.client = client

            def load_model(self) -> Any:
                return self.client

            def generate(self, prompt: str) -> str:
                return self.client.complete([ChatMessage("user", prompt)]).text

            async def a_generate(self, prompt: str) -> str:
                return self.generate(prompt)

            def get_model_name(self) -> str:
                return self.cfg.generate.model

        return _Wrapper()

    def evaluate(self, cases: Sequence[CaseSpec], results: Sequence[CaseResult]) -> dict[str, dict[str, float]]:
        if not self.available:
            return {}
        try:  # pragma: no cover - 需真实依赖
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase
        except Exception as exc:
            self.status.reason = f"DeepEval 导入失败：{exc}"
            self.status.available = False
            return {}
        model = None
        try:
            model = self._model()
        except AdapterUnavailable as exc:
            self.status.reason = str(exc)
            return {}

        mapping: dict[str, dict[str, float]] = {}
        for case, result in zip(cases, results):  # pragma: no cover
            test_case = LLMTestCase(
                input=case.question,
                actual_output=result.answer,
                expected_output=case.reference_answer or "",
                retrieval_context=[item.chunk.text for item in result.contexts],
            )
            row: dict[str, float] = {}
            for name, metric_cls, key in (
                ("faithfulness", FaithfulnessMetric, "faithfulness"),
                ("answer_relevancy", AnswerRelevancyMetric, "answer_relevancy"),
                ("contextual_precision", ContextualPrecisionMetric, "context_precision"),
                ("contextual_recall", ContextualRecallMetric, "context_recall"),
            ):
                if case.should_refuse and name in ("contextual_precision", "contextual_recall"):
                    continue
                try:
                    metric = metric_cls(model=model, threshold=0.5, include_reason=False)
                    metric.measure(test_case)
                    if metric.score is not None:
                        row[key] = float(metric.score)
                except Exception:
                    continue
            if row:
                mapping[case.case_id] = row
        return mapping


def adapter_statuses(cfg: Config) -> list[dict[str, Any]]:
    return [ragas_status(cfg).to_dict(), deepeval_status(cfg).to_dict()]
