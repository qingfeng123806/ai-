"""Judge 维度的指标暴露：把 LLM-as-Judge 的评分接入统一指标注册表。

这些指标与原生指标并列展示，用于交叉校验：若 ``judge_faithfulness`` 明显高于
原生 ``faithfulness``，说明词法支撑判定过松（或 judge 过宽），值得复核。
未启用 judge 的用例会返回 ``None``，聚合时自动跳过，不会污染均值。
"""

from __future__ import annotations

from ..types import CaseResult
from .base import EvalContext, MetricSpec, case_metric, clamp01


def _value(ctx: EvalContext, key: str) -> float | None:
    value = ctx.judge_scores.get(key)
    return None if value is None else clamp01(value)


@case_metric(
    MetricSpec(
        name="judge_faithfulness",
        group="judge",
        description="LLM-as-Judge 给出的忠实度评分（无 LLM 时为启发式量规）",
    )
)
def judge_faithfulness(result: CaseResult, ctx: EvalContext) -> float | None:
    return _value(ctx, "faithfulness")


@case_metric(
    MetricSpec(
        name="judge_relevance",
        group="judge",
        description="LLM-as-Judge 给出的相关性评分",
    )
)
def judge_relevance(result: CaseResult, ctx: EvalContext) -> float | None:
    return _value(ctx, "relevance")


@case_metric(
    MetricSpec(
        name="judge_completeness",
        group="judge",
        description="LLM-as-Judge 给出的完整性评分",
    )
)
def judge_completeness(result: CaseResult, ctx: EvalContext) -> float | None:
    return _value(ctx, "completeness")


@case_metric(
    MetricSpec(
        name="judge_citation_quality",
        group="judge",
        description="LLM-as-Judge 给出的引用质量评分",
    )
)
def judge_citation_quality(result: CaseResult, ctx: EvalContext) -> float | None:
    return _value(ctx, "citation_quality")


@case_metric(
    MetricSpec(
        name="judge_overall",
        group="judge",
        description="Judge 四个维度的均值，作为综合主观质量分",
    )
)
def judge_overall(result: CaseResult, ctx: EvalContext) -> float | None:
    values = [ctx.judge_scores.get(k) for k in ("faithfulness", "relevance", "completeness", "citation_quality")]
    present = [v for v in values if v is not None]
    if not present:
        return None
    return clamp01(sum(present) / len(present))
