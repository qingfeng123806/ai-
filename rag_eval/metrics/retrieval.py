"""检索质量指标：命中率、精确率、召回率、MRR、nDCG、上下文精确率/召回率、冗余度。"""

from __future__ import annotations

import math
from typing import Sequence

from ..text import coverage, jaccard, term_set
from ..types import CaseResult, Retrieved
from .base import EvalContext, case_metric, chunk_relevance, clamp01, MetricSpec


def _not_applicable(result: CaseResult, ctx: EvalContext) -> bool:
    """对抗集中应拒答的用例，检索相关性没有评判意义（参考答案是「拒绝」而非某片段）。"""
    return bool(ctx.case.should_refuse)

def _relevance_flags(result: CaseResult, ctx: EvalContext) -> list[bool]:
    return [chunk_relevance(item, ctx.case, ctx.idf) for item in result.contexts]


@case_metric(
    MetricSpec(
        name="hit_rate",
        group="retrieval",
        description="TopK 结果中是否至少有一个与参考答案相关（二值命中率）",
    )
)
def hit_rate(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    if not result.contexts:
        return 0.0
    return 1.0 if any(_relevance_flags(result, ctx)) else 0.0


@case_metric(
    MetricSpec(
        name="precision_at_k",
        group="retrieval",
        description="TopK 片段中相关片段的比例（检索精确率）",
    )
)
def precision_at_k(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    if not result.contexts:
        return 0.0
    flags = _relevance_flags(result, ctx)
    return clamp01(sum(flags) / len(flags))


@case_metric(
    MetricSpec(
        name="recall_at_k",
        group="retrieval",
        description="参考上下文/要点被 TopK 片段覆盖的比例（检索召回率）",
    )
)
def recall_at_k(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    joined = [item.chunk.text for item in result.contexts]
    if not joined:
        return 0.0
    case = ctx.case
    if case.reference_contexts:
        hits = 0
        for ref in case.reference_contexts:
            ref_terms = term_set(ref)
            if not ref_terms:
                continue
            if any(coverage(ref_terms, term_set(text), ctx.idf) >= 0.55 for text in joined):
                hits += 1
        return clamp01(hits / max(1, len(case.reference_contexts)))
    if case.reference_answer:
        return clamp01(coverage(term_set(case.reference_answer), set().union(*(term_set(t) for t in joined))))
    if case.expected_keywords:
        blob = " ".join(joined)
        return clamp01(sum(1 for kw in case.expected_keywords if kw.lower() in blob.lower()) / len(case.expected_keywords))
    return 1.0 if joined else 0.0


@case_metric(
    MetricSpec(
        name="mrr",
        group="retrieval",
        description="第一个相关片段的名次倒数（Mean Reciprocal Rank）",
    )
)
def mrr(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    for rank, flag in enumerate(_relevance_flags(result, ctx), start=1):
        if flag:
            return 1.0 / rank
    return 0.0


@case_metric(
    MetricSpec(
        name="ndcg",
        group="retrieval",
        description="按二值相关性计算的归一化折损累计增益",
    )
)
def ndcg(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    flags = _relevance_flags(result, ctx)
    if not flags:
        return 0.0
    dcg = sum((1.0 if flag else 0.0) / math.log2(i + 2) for i, flag in enumerate(flags))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(sum(flags)))
    return clamp01(dcg / ideal if ideal else 0.0)


@case_metric(
    MetricSpec(
        name="context_precision",
        group="retrieval",
        description="RAGAs 口径：按位置加权的相关片段精确率（越靠前越重要）",
    )
)
def context_precision(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    flags = _relevance_flags(result, ctx)
    if not flags:
        return 0.0
    total = 0.0
    hits = 0
    for i, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            total += hits / i
    return clamp01(total / len(flags))


@case_metric(
    MetricSpec(
        name="context_recall",
        group="retrieval",
        description="RAGAs 口径：参考要点被检索上下文覆盖的比例",
    )
)
def context_recall(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    if not result.contexts:
        return 0.0
    context_terms = set().union(*(term_set(item.chunk.text) for item in result.contexts))
    if ctx.case.reference_answer:
        return clamp01(coverage(term_set(ctx.case.reference_answer), context_terms, ctx.idf))
    if ctx.case.expected_keywords:
        blob = " ".join(item.chunk.text for item in result.contexts)
        hit = sum(1 for kw in ctx.case.expected_keywords if kw.lower() in blob.lower())
        return clamp01(hit / max(1, len(ctx.case.expected_keywords)))
    return 1.0


@case_metric(
    MetricSpec(
        name="context_redundancy",
        group="retrieval",
        direction="lower",
        description="检索片段之间的平均重复度（越高表示越同质，浪费上下文预算）",
    )
)
def context_redundancy(result: CaseResult, ctx: EvalContext) -> float:
    if _not_applicable(result, ctx):
        return None
    texts = [item.chunk.text for item in result.contexts]
    if len(texts) < 2:
        return 0.0
    sims: list[float] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(jaccard(term_set(texts[i]), term_set(texts[j])))
    return clamp01(sum(sims) / len(sims) if sims else 0.0)
