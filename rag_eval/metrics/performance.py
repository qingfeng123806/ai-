"""性能与成本指标（聚合口径）：延迟分位、Token 用量、单次查询成本、错误率。"""

from __future__ import annotations

from typing import Sequence

from ..types import CaseResult
from .base import MetricSpec, agg_metric, clamp01


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


@agg_metric(
    MetricSpec(
        name="latency_p50",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="端到端延迟中位数（毫秒）",
        unit="ms",
    )
)
def latency_p50(results: Sequence[CaseResult]) -> float:
    return _percentile([r.latency.total_ms for r in results], 0.50)


@agg_metric(
    MetricSpec(
        name="latency_p95",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="端到端延迟 P95（毫秒）",
        unit="ms",
    )
)
def latency_p95(results: Sequence[CaseResult]) -> float:
    return _percentile([r.latency.total_ms for r in results], 0.95)


@agg_metric(
    MetricSpec(
        name="latency_retrieval_p95",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="检索 + 重排序阶段 P95 延迟（毫秒）",
        unit="ms",
    )
)
def latency_retrieval_p95(results: Sequence[CaseResult]) -> float:
    return _percentile([r.latency.retrieval_ms + r.latency.rerank_ms for r in results], 0.95)


@agg_metric(
    MetricSpec(
        name="tokens_per_answer",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="单次问答平均 Token 消耗（输入 + 输出）",
        unit="tokens",
    )
)
def tokens_per_answer(results: Sequence[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(r.tokens_in + r.tokens_out for r in results) / len(results)


@agg_metric(
    MetricSpec(
        name="cost_per_query",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="单次问答平均成本（美元，按配置单价核算）",
        unit="usd",
    )
)
def cost_per_query(results: Sequence[CaseResult]) -> float:
    if not results:
        return 0.0
    return sum(r.cost_usd for r in results) / len(results)


@agg_metric(
    MetricSpec(
        name="error_rate",
        group="performance",
        scope="aggregate",
        direction="lower",
        description="执行失败（异常或空答案）的用例比例",
    )
)
def error_rate(results: Sequence[CaseResult]) -> float:
    if not results:
        return 0.0
    bad = sum(1 for r in results if r.error or not r.answer.strip())
    return clamp01(bad / len(results))


@agg_metric(
    MetricSpec(
        name="refusal_rate",
        group="safety",
        scope="aggregate",
        direction="lower",
        description="整体拒答率（需结合拒答恰当性一起解读，过低易幻觉、过高易答非所答）",
    )
)
def refusal_rate(results: Sequence[CaseResult]) -> float:
    if not results:
        return 0.0
    return clamp01(sum(1 for r in results if r.refused) / len(results))
