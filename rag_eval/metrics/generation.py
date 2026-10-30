"""生成质量与引用质量指标：忠实度、答案相关性、完整性、幻觉率、重复度、引用准确率。

忠实度的实现是「逐句可支撑性」：把答案拆成句子，检查每句的 idf 加权词覆盖能否
在其中某个检索片段里得到支撑；数字必须精确出现才视为支撑（数字幻觉最容易验证）。
这与 RAGAs 的 statement-level entailment 思路一致，只是用词法蕴含替代 LLM 判定，
因此在离线环境可用，且给出可解释的逐句证据。
"""

from __future__ import annotations

import re
from typing import Sequence

from ..text import (
    extract_citations,
    is_refusal,
    jaccard,
    sentences,
    strip_citations,
    term_set,
)
from ..types import CaseResult
from .base import (
    EvalContext,
    case_metric,
    clamp01,
    is_supported,
    keyword_score,
    sentence_support,
    MetricSpec,
)

_REFUSAL_ANSWER = "（拒答，无需引用）"


def _content_sentences(answer: str) -> list[str]:
    cleaned = strip_citations(answer)
    sents = [s for s in sentences(cleaned) if len(term_set(s)) >= 2]
    return sents


def _is_refusal_case(result: CaseResult) -> bool:
    return result.refused or is_refusal(result.answer)


@case_metric(
    MetricSpec(
        name="faithfulness",
        group="generation",
        description="答案中每句话能在检索上下文中找到支撑的比例（忠实度）",
    )
)
def faithfulness(result: CaseResult, ctx: EvalContext) -> float:
    if not result.answer.strip():
        return 0.0
    if _is_refusal_case(result):
        return 1.0  # 拒答不产生事实性断言，不存在不忠实
    sents = _content_sentences(result.answer)
    if not sents:
        return 1.0
    total = 0.0
    hit = 0.0
    for sent in sents:
        weight = max(1.0, sum(ctx.weight(t) for t in term_set(sent)))
        total += weight
        if is_supported(sent, result.contexts, ctx.idf):
            hit += weight
    return clamp01(hit / total if total else 0.0)


@case_metric(
    MetricSpec(
        name="answer_relevancy",
        group="generation",
        description="答案与问题的语义贴合度（问题词覆盖 + 问题类型是否被回应 + 信息量）",
    )
)
def answer_relevancy(result: CaseResult, ctx: EvalContext) -> float:
    if not result.answer.strip():
        return 0.0
    if ctx.case.should_refuse and _is_refusal_case(result):
        return 1.0  # 该拒答时拒答，就是最相关的响应
    if _is_refusal_case(result):
        return 0.0
    from ..text import coverage

    q_terms = term_set(ctx.case.question)
    answer = strip_citations(result.answer)
    a_terms = term_set(answer)
    base = coverage(q_terms, a_terms, ctx.idf)
    # 问题问了价格/时限/数量这类「可验证类型」时，答案是否给出对应类型的值，
    # 比单纯的词汇重合更能反映相关性（例如价格题的答案是一行报价表）。
    type_match = _type_match(ctx.case.question, answer)
    length_factor = min(1.0, len(answer) / max(24.0, 0.6 * len(ctx.case.question)))
    return clamp01(0.55 * base + 0.30 * type_match + 0.15 * length_factor)


_ANSWER_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"价格|收费|多少钱|费用|报价|计费", ("元", "￥", "价格", "费用", "计费")),
    (r"多久|时限|多长时间|几天|几小时|保留", ("分钟", "小时", "天", "秒", "工作日", "年")),
    (r"多少|几|上限|最多|多大", ("人", "条", "GB", "个", "次", "元", "分钟", "小时", "天")),
)


def _type_match(question: str, answer: str) -> float:
    for pattern, markers in _ANSWER_TYPES:
        if re.search(pattern, question):
            return 1.0 if any(marker in answer for marker in markers) else 0.0
    return 0.5  # 非可验证类型问题：不奖不罚


@case_metric(
    MetricSpec(
        name="answer_completeness",
        group="generation",
        description="参考答案要点/关键词在答案中的覆盖率（完整性）",
    )
)
def answer_completeness(result: CaseResult, ctx: EvalContext) -> float:
    if ctx.case.should_refuse:
        return 1.0 if _is_refusal_case(result) else 0.0
    if not result.answer.strip() or _is_refusal_case(result):
        return 0.0
    return clamp01(keyword_score(ctx.case, strip_citations(result.answer)))


@case_metric(
    MetricSpec(
        name="hallucination_rate",
        group="generation",
        direction="lower",
        description="答案中无法被上下文支撑的句子占比（幻觉率）",
    )
)
def hallucination_rate(result: CaseResult, ctx: EvalContext) -> float:
    if not result.answer.strip():
        return 0.0
    if _is_refusal_case(result):
        return 0.0
    sents = _content_sentences(result.answer)
    if not sents:
        return 0.0
    unsupported = sum(1 for sent in sents if not is_supported(sent, result.contexts, ctx.idf))
    return clamp01(unsupported / len(sents))


@case_metric(
    MetricSpec(
        name="answer_repetition",
        group="generation",
        direction="lower",
        description="答案内部近似重复句子的比例（重复回答问题）",
    )
)
def answer_repetition(result: CaseResult, ctx: EvalContext) -> float:
    sents = _content_sentences(result.answer)
    if len(sents) < 2:
        return 0.0
    duplicates = 0
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            if jaccard(term_set(sents[i]), term_set(sents[j])) >= 0.85:
                duplicates += 1
                break
    return clamp01(duplicates / len(sents))


# --------------------------------------------------------------------------
# 引用质量
# --------------------------------------------------------------------------

def _citation_checks(result: CaseResult, ctx: EvalContext) -> tuple[int, int, int]:
    """返回 (引用总数, 引用正确数, 带引用的内容句数)。"""
    if not result.answer.strip() or _is_refusal_case(result):
        return 0, 0, 0
    total = 0
    correct = 0
    sentences_with_cite = 0
    for raw_sent in sentences(result.answer):
        markers = extract_citations(raw_sent)
        plain = strip_citations(raw_sent)
        if len(term_set(plain)) < 2:
            continue
        if markers:
            sentences_with_cite += 1
        for marker in markers:
            total += 1
            if marker < 1 or marker > len(result.contexts):
                continue
            support = sentence_support(plain, [result.contexts[marker - 1]], ctx.idf)
            if support >= 0.5:
                correct += 1
    return total, correct, sentences_with_cite


@case_metric(
    MetricSpec(
        name="citation_accuracy",
        group="citation",
        description="引用是否指向真正支撑该句的片段（引用准确率）",
    )
)
def citation_accuracy(result: CaseResult, ctx: EvalContext) -> float | None:
    if _is_refusal_case(result):
        return 1.0
    total, correct, _ = _citation_checks(result, ctx)
    if total == 0:
        return 0.0
    return clamp01(correct / total)


@case_metric(
    MetricSpec(
        name="citation_coverage",
        group="citation",
        description="有事实性断言的句子中带引用的比例（引用覆盖率）",
    )
)
def citation_coverage(result: CaseResult, ctx: EvalContext) -> float:
    if _is_refusal_case(result):
        return 1.0
    sents = _content_sentences(result.answer)
    if not sents:
        return 0.0
    _, _, cited = _citation_checks(result, ctx)
    return clamp01(cited / len(sents))


@case_metric(
    MetricSpec(
        name="reference_present",
        group="citation",
        description="答案是否附带了至少一条可回溯的引用（二值）",
    )
)
def reference_present(result: CaseResult, ctx: EvalContext) -> float:
    if _is_refusal_case(result):
        return 1.0
    return 1.0 if result.citations else 0.0
