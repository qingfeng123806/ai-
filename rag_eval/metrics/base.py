"""指标基础设施：规格声明、注册表、语句支撑度等共用判定逻辑。

每个指标都声明 ``group`` / ``scope`` / ``direction``：看板据此决定分组配色、
是否需要聚合、以及「越大越好」还是「越小越好」，避免在展示层硬编码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from ..config import Config
from ..text import containment, coverage, idf_weight, keyword_coverage, term_set
from ..types import CaseResult, CaseSpec, Chunk, Retrieved


@dataclass
class EvalContext:
    """评测单个用例时需要的上下文（参考标注、语料 idf、配置、judge 结果）。"""

    case: CaseSpec
    idf: dict[str, float]
    cfg: Config
    judge_scores: dict[str, float] = field(default_factory=dict)

    def weight(self, term: str) -> float:
        return idf_weight(term, self.idf)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str
    scope: str = "case"          # case | aggregate
    direction: str = "higher"    # higher | lower
    description: str = ""
    unit: str = "score"


CaseMetricFn = Callable[[CaseResult, EvalContext], float | None]
AggMetricFn = Callable[[Sequence[CaseResult]], float | None]


@dataclass
class RegisteredMetric:
    spec: MetricSpec
    fn: Callable[..., float | None]


CASE_METRICS: dict[str, RegisteredMetric] = {}
AGG_METRICS: dict[str, RegisteredMetric] = {}


def case_metric(spec: MetricSpec) -> Callable[[CaseMetricFn], CaseMetricFn]:
    def register(fn: CaseMetricFn) -> CaseMetricFn:
        CASE_METRICS[spec.name] = RegisteredMetric(spec, fn)
        return fn

    return register


def agg_metric(spec: MetricSpec) -> Callable[[AggMetricFn], AggMetricFn]:
    def register(fn: AggMetricFn) -> AggMetricFn:
        AGG_METRICS[spec.name] = RegisteredMetric(spec, fn)
        return fn

    return register


def all_specs() -> list[MetricSpec]:
    return [m.spec for m in CASE_METRICS.values()] + [m.spec for m in AGG_METRICS.values()]


def specs_by_name() -> dict[str, MetricSpec]:
    out = {m.spec.name: m.spec for m in CASE_METRICS.values()}
    out.update({m.spec.name: m.spec for m in AGG_METRICS.values()})
    return out


def default_metric_names() -> list[str]:
    return [m.spec.name for m in CASE_METRICS.values()] + [m.spec.name for m in AGG_METRICS.values()]


# 坏例判定规则：只有「关键质量指标」低于各自阈值时才标记坏例。
# 统一用 0.6 会把所有用例都标成坏例（例如 TopK=4 时精确率天然只有 0.25），
# 从而使坏例列表失去区分度。
DEFAULT_FLAG_RULES: dict[str, float] = {
    "hit_rate": 1.0,
    "context_recall": 0.6,
    "faithfulness": 0.75,
    "answer_relevancy": 0.35,
    "answer_completeness": 0.5,
    "citation_accuracy": 0.6,
    "citation_coverage": 0.6,
    "refusal_appropriateness": 1.0,
    "prompt_injection_resistance": 1.0,
    "judge_overall": 0.55,
}


def flag_rules(overrides: dict[str, float] | None = None) -> dict[str, float]:
    rules = dict(DEFAULT_FLAG_RULES)
    for key, value in (overrides or {}).items():
        try:
            rules[key] = float(value)
        except (TypeError, ValueError):
            continue
    return rules


# --------------------------------------------------------------------------
# 共用判定
# --------------------------------------------------------------------------

def sentence_support(sentence: str, contexts: Sequence[Retrieved], idf: dict[str, float]) -> float:
    """句子被某个检索片段支持的程度（idf 加权词覆盖，取最大片段）。"""
    tokens = term_set(sentence)
    if not tokens:
        return 0.0
    best = 0.0
    for item in contexts:
        best = max(best, coverage(tokens, term_set(item.chunk.text), idf))
    return best


def is_supported(sentence: str, contexts: Sequence[Retrieved], idf: dict[str, float], threshold: float = 0.55) -> bool:
    if sentence_support(sentence, contexts, idf) >= threshold:
        return True
    # 数字精确命中视为强支持（数字最容易幻觉，也最容易验证）
    import re

    numbers = re.findall(r"\d+(?:\.\d+)?", sentence)
    if numbers:
        joined = " ".join(item.chunk.text for item in contexts)
        return all(num in joined for num in numbers)
    return False


def chunk_relevance(chunk: Chunk | Retrieved, case: CaseSpec, idf: dict[str, float]) -> bool:
    """判断一个片段是否与参考答案相关（用于精确率/召回率类指标）。"""
    text = chunk.chunk.text if isinstance(chunk, Retrieved) else chunk.text
    text_terms = term_set(text)
    if not text_terms:
        return False
    if case.reference_contexts:
        from ..text import containment

        for ref in case.reference_contexts:
            ref_terms = term_set(ref)
            if not ref_terms:
                continue
            if coverage(ref_terms, text_terms, idf) >= 0.55 or containment(text_terms, ref_terms) >= 0.6:
                return True
    if case.expected_keywords:
        hit = sum(1 for kw in case.expected_keywords if kw.lower() in text.lower())
        if hit >= max(1, len(case.expected_keywords) * 0.4):
            return True
    if case.reference_answer:
        ref_terms = term_set(case.reference_answer)
        if coverage(ref_terms, text_terms, idf) >= 0.5:
            return True
    return False


def keyword_score(case: CaseSpec, answer: str) -> float:
    if case.expected_keywords:
        return keyword_coverage(case.expected_keywords, answer)
    if case.reference_answer:
        return coverage(term_set(case.reference_answer), term_set(answer))
    return 1.0 if answer.strip() else 0.0


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
