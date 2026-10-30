"""指标注册与评测调度：指标计算、坏例标记、聚合与置信区间。"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..stats import bootstrap_ci, mean
from ..text import extract_citations, is_refusal, strip_citations, term_set
from ..types import CaseResult, MetricScore
from . import generation, judge_metrics, performance, retrieval, safety  # noqa: F401  注册副作用
from .base import (
    AGG_METRICS,
    CASE_METRICS,
    DEFAULT_FLAG_RULES,
    EvalContext,
    MetricSpec,
    all_specs,
    chunk_relevance,
    default_metric_names,
    flag_rules,
    is_supported,
    sentence_support,
    specs_by_name,
)

QUALITY_GROUPS = {"retrieval", "generation", "citation", "safety", "judge"}

__all__ = [
    "EvalContext",
    "MetricSpec",
    "CASE_METRICS",
    "AGG_METRICS",
    "DEFAULT_FLAG_RULES",
    "all_specs",
    "specs_by_name",
    "default_metric_names",
    "available_metric_names",
    "validate_metric_names",
    "evaluate_case",
    "aggregate_metrics",
    "metric_series",
    "explain_case",
]


def available_metric_names() -> list[str]:
    return default_metric_names()


def validate_metric_names(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """返回 (有效指标, 未知指标)，未知项不应静默忽略。"""
    known = specs_by_name()
    valid = [n for n in names if n in known]
    unknown = [n for n in names if n not in known]
    return valid, unknown


def explain_case(result: CaseResult, ctx: EvalContext) -> dict[str, Any]:
    """生成逐句证据与片段相关性，供 badcase 详情页展示。"""
    supports: list[dict[str, Any]] = []
    if not (result.refused or is_refusal(result.answer)):
        from ..text import sentences

        for sent in sentences(result.answer):
            plain = strip_citations(sent)
            if len(term_set(plain)) < 2:
                continue
            support = sentence_support(plain, result.contexts, ctx.idf)
            supports.append(
                {
                    "sentence": plain,
                    "citations": extract_citations(sent),
                    "support": round(support, 3),
                    "supported": is_supported(plain, result.contexts, ctx.idf),
                }
            )
    relevance = [
        {
            "chunk_id": item.chunk.chunk_id,
            "title": item.chunk.title,
            "rank": item.rank,
            "score": round(item.score, 4),
            "relevant": chunk_relevance(item, ctx.case, ctx.idf),
        }
        for item in result.contexts
    ]
    return {"support": supports, "relevance": relevance}


def evaluate_case(
    result: CaseResult,
    ctx: EvalContext,
    metric_names: Sequence[str],
    flag_rules_cfg: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """计算单个用例的全部指标，并按关键指标阈值标记坏例。"""
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name in metric_names:
        registered = CASE_METRICS.get(name)
        if registered is None:
            continue
        try:
            value = registered.fn(result, ctx)
        except Exception as exc:  # 单个指标异常不应中断整次评测
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if value is None:
            continue
        scores[name] = round(max(0.0, min(1.0, float(value))), 4)

    rules = flag_rules(flag_rules_cfg)
    flags: list[str] = []
    for name, threshold in rules.items():
        value = scores.get(name)
        if value is None:
            continue
        spec = CASE_METRICS.get(name)
        if spec is None or spec.spec.group not in QUALITY_GROUPS:
            continue
        quality = value if spec.spec.direction == "higher" else 1.0 - value
        if quality < threshold:
            flags.append(name)

    details = explain_case(result, ctx)
    details.update(
        {
            "flags": flags,
            "flagged": bool(flags),
            "metric_errors": errors,
        }
    )
    return scores, details


def metric_series(results: Sequence[CaseResult], name: str) -> list[float]:
    """取逐用例指标序列（用于配对显著性检验）；缺失值按跳过处理。"""
    return [float(r.scores[name]) for r in results if name in r.scores]


def _bootstrap_aggregate(
    results: Sequence[CaseResult],
    fn,
    samples: int,
    confidence: float,
    seed: str,
) -> tuple[float, float]:
    """对聚合指标（P95、均值等）做 Bootstrap：重采样用例后重新计算指标。"""
    import random

    if samples <= 0 or len(results) < 3:
        value = fn(results)
        return (float(value or 0.0), float(value or 0.0))
    rng = random.Random(seed)
    n = len(results)
    values: list[float] = []
    for _ in range(samples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        try:
            values.append(float(fn(sample)))
        except Exception:
            continue
    if not values:
        value = fn(results) or 0.0
        return float(value), float(value)
    values.sort()
    alpha = (1.0 - confidence) / 2.0
    low = values[int(alpha * (len(values) - 1))]
    high = values[int((1 - alpha) * (len(values) - 1))]
    return low, high


def aggregate_metrics(
    results: Sequence[CaseResult],
    metric_names: Sequence[str],
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int | str = 0,
) -> list[MetricScore]:
    """把逐用例结果汇总为运行级指标（含 Bootstrap 置信区间）。"""
    out: list[MetricScore] = []
    specs = specs_by_name()
    for name in metric_names:
        spec = specs.get(name)
        if spec is None:
            continue
        if name in AGG_METRICS and spec.scope == "aggregate":
            fn = AGG_METRICS[name].fn
            value = fn(results)
            if value is None:
                continue
            n = len(results)
            low, high = _bootstrap_aggregate(
                results, fn, bootstrap_samples, confidence, f"{seed}:{name}"
            )
        else:
            values = metric_series(results, name)
            if not values:
                continue
            value = mean(values)
            n = len(values)
            low, high = bootstrap_ci(values, samples=bootstrap_samples, confidence=confidence, seed=f"{seed}:{name}")
        out.append(
            MetricScore(
                name=name,
                value=float(value),
                group=spec.group,
                scope=spec.scope,
                direction=spec.direction,
                n=n,
                ci_low=float(low),
                ci_high=float(high),
                description=spec.description,
            )
        )
    return out


def group_metrics(metrics: Sequence[MetricScore]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(metric.group, []).append(metric.to_dict())
    return grouped
