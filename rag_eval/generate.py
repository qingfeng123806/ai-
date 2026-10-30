"""答案生成：Prompt 模板注册表 + 引用约束 + 离线抽取式生成器。

Prompt 是可被 A/B 实验替换的一等公民：每个模板声明 ``style``（句数上限、
拒答阈值等），在线后端把 style 写进指令，离线后端则按 style 近似执行，
使「换 Prompt」的实验在两种模式下都产生可比的行为差异。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import GenerateCfg
from .llm import ChatMessage, LLMClient, LLMError, LLMResponse, estimate_tokens
from .text import (
    build_idf,
    coverage,
    dedupe_sentences,
    detect_injection,
    extract_citations,
    idf_weight,
    is_refusal,
    jaccard,
    keyword_coverage,
    normalize_ws,
    sentences,
    term_set,
    terms,
    truncate,
)
from .types import Citation, Generation, Retrieved

CITATION_RULE = (
    "引用规则：每个事实性论断后必须紧跟对应的片段编号，例如 [1]；"
    "一句话可以引用多个片段，写作 [1][3]。不得编造编号。"
)

REFUSAL_RULE = (
    "如果参考片段不足以回答问题，直接说明「根据现有知识库无法回答该问题」，"
    "并指出缺少哪类信息；绝对不要依据常识自行补充或编造。"
)

SYSTEM_BASE = (
    "你是企业知识库问答助手。你只能依据给定的参考片段作答，"
    "不得使用片段之外的知识，不得推测未给出的事实。"
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str
    style: dict[str, Any] = field(default_factory=dict)

    def render(self, question: str, context_block: str) -> tuple[str, str]:
        system = self.system.strip()
        user = self.user.replace("{question}", question).replace("{context}", context_block).strip()
        return system, user


PROMPTS: dict[str, PromptTemplate] = {
    "grounded_cite_v1": PromptTemplate(
        name="grounded_cite_v1",
        system=SYSTEM_BASE + "回答使用简体中文，先给结论再给要点。" + CITATION_RULE + REFUSAL_RULE,
        user=(
            "参考片段：\n{context}\n\n"
            "用户问题：{question}\n\n"
            "请作答：\n"
            "1) 只使用参考片段中的信息；\n"
            "2) 每个事实后紧跟 [编号] 引用；\n"
            "3) 涉及数字、价格、时限必须原样保留；\n"
            "4) 信息不足时明确拒答。"
        ),
        style={"max_sentences": 4, "min_score": 0.28, "require_citation": True},
    ),
    "concise_v1": PromptTemplate(
        name="concise_v1",
        system=SYSTEM_BASE + "回答必须简洁，最多 3 句，直接给结论，不要展开背景。" + CITATION_RULE + REFUSAL_RULE,
        user=(
            "参考片段：\n{context}\n\n"
            "用户问题：{question}\n\n"
            "请用不超过 3 句直接回答，句末标注引用编号。"
        ),
        style={"max_sentences": 2, "min_score": 0.30, "require_citation": True},
    ),
    "strict_v2": PromptTemplate(
        name="strict_v2",
        system=SYSTEM_BASE
        + "采取最保守策略：只要片段中没有直接、完整的依据，就必须拒答，宁可少答不可错答。"
        + CITATION_RULE
        + REFUSAL_RULE,
        user=(
            "参考片段：\n{context}\n\n"
            "用户问题：{question}\n\n"
            "请先判断片段是否包含直接依据：\n"
            "- 若包含，给出结论并标注引用；\n"
            "- 若不包含，只回复「根据现有知识库无法回答该问题」并说明缺少的信息。"
        ),
        style={"max_sentences": 3, "min_score": 0.60, "require_citation": True},
    ),
}


def get_prompt(name: str) -> PromptTemplate:
    if name in PROMPTS:
        return PROMPTS[name]
    raise KeyError(f"未知 Prompt 模板 {name}，可用：{sorted(PROMPTS)}")


_CONTEXT_HEADER = re.compile(r"^【[^】]{0,80}】\s*")
_MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def format_contexts(contexts: Sequence[Retrieved], max_chars: int = 4000) -> str:
    """把检索结果渲染成编号片段块，编号与引用 [n] 一一对应。"""
    blocks: list[str] = []
    used = 0
    for i, item in enumerate(contexts, start=1):
        text = item.chunk.text
        header = f"[{i}] 《{item.chunk.title}》"
        headings = item.chunk.meta.get("headings") or []
        if headings:
            header += " / " + " / ".join(headings[-2:])
        body = truncate(text, max(200, max_chars // max(1, len(contexts))))
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _clean_sentence(text: str) -> str:
    text = _CONTEXT_HEADER.sub("", text.strip())
    text = re.sub(r"^[-*+]\s+", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text)
    return text.strip().strip("*").strip()


def _is_evidence_sentence(cleaned: str, raw: str) -> bool:
    """过滤标题行、表格分隔线、FAQ 问句等「像句子但不含事实」的片段。"""
    if len(cleaned) < 6 or len(cleaned) > 400:
        return False
    if raw.lstrip().startswith("#"):
        return False
    if re.fullmatch(r"[\s|:\-+*=#`~_]+", cleaned):
        return False
    if re.match(r"^\|?[\s:\-|]+\|?$", cleaned):
        return False
    if _is_table_header(cleaned):
        return False
    if re.match(r"^\**Q\s*[:：]", cleaned):
        return False
    if cleaned.endswith(("：", ":")) and len(cleaned) < 40:
        return False
    return True


def _is_table_header(sentence: str) -> bool:
    """Markdown 表头：单元格都很短且不含数字（表头行本身通常没有事实值）。"""
    if sentence.count("|") < 3:
        return False
    cells = [c.strip() for c in sentence.strip("|").split("|") if c.strip()]
    if len(cells) < 3:
        return False
    return all(len(c) <= 6 and not re.search(r"\d", c) for c in cells)


# 问题类型 → 答案中应出现的线索词，用于抑制「同主题但答非所问」的句子
_ANSWER_TYPE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"价格|收费|多少钱|费用|报价|计费", ("元", "￥", "价格", "费用", "计费", "报价")),
    (r"多久|时限|多长时间|几天|几分钟|几小时|保留", ("分钟", "小时", "天", "秒", "工作日", "年")),
    (r"多少人|人数|上限|最多|几条|多大", ("人", "条", "GB", "个", "次")),
)


def _numeric_terms(question: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", question)


@dataclass
class Answerer:
    """统一入口：在线走 LLM，离线走向量/词法打分的抽取式生成。"""

    cfg: GenerateCfg
    llm: LLMClient | None = None
    corpus_idf: dict[str, float] | None = None
    _prompt: PromptTemplate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._prompt = get_prompt(self.cfg.prompt)

    # -- 元信息 ----------------------------------------------------------
    @property
    def mode(self) -> str:
        return "live" if self.llm is not None else "offline"

    def info(self) -> dict[str, Any]:
        return {
            "provider": "openai" if self.llm is not None else "offline-extractive",
            "model": getattr(self.llm, "model", "extractive-v1") if self.llm is not None else "extractive-v1",
            "prompt": self.cfg.prompt,
            "mode": self.mode,
        }

    # -- 主流程 ----------------------------------------------------------
    def answer(self, question: str, contexts: Sequence[Retrieved]) -> Generation:
        if self.llm is None:
            return self._offline(question, contexts)
        try:
            return self._online(question, contexts)
        except LLMError as exc:
            generation = self._offline(question, contexts)
            generation.provider = "offline-extractive(fallback)"
            generation.raw = {"error": str(exc), "fallback": True}
            return generation

    def _online(self, question: str, contexts: Sequence[Retrieved]) -> Generation:
        block = format_contexts(contexts, self.cfg.max_context_chars)
        system, user = self._prompt.render(question, block)
        messages = [ChatMessage("system", system), ChatMessage("user", user)]
        response: LLMResponse = self.llm.complete(  # type: ignore[union-attr]
            messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        answer = response.text.strip()
        if self.cfg.dedupe_sentences:
            answer = dedupe_sentences(answer)
        parsed = self._attach_citations(answer, contexts)
        refused = is_refusal(answer) or (not parsed and not answer)
        return Generation(
            answer=answer,
            citations=parsed,
            refused=refused,
            model=response.model,
            provider=response.provider,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            raw=response.raw,
        )

    def _offline(self, question: str, contexts: Sequence[Retrieved]) -> Generation:
        style = self._prompt.style
        max_sentences = int(style.get("max_sentences", 4))
        min_score = float(style.get("min_score", 0.10))

        injection = detect_injection(question)
        if injection:
            text = (
                "无法执行该请求：该问题包含试图覆盖系统指令或获取内部信息的内容（"
                + "、".join(sorted(set(injection)))
                + "）。请提出与云雀协作产品相关的正常问题。"
            )
            return Generation(answer=text, refused=True, model="extractive-v1", provider="offline-extractive",
                              raw={"guard": "injection", "hits": injection})

        if not contexts:
            return Generation(
                answer="根据现有知识库无法回答该问题：未检索到相关片段。",
                refused=True,
                model="extractive-v1",
                provider="offline-extractive",
            )

        corpus_idf = self.corpus_idf or build_idf([item.chunk.text for item in contexts])
        q_terms = term_set(question)
        q_numbers = _numeric_terms(question)
        asks_number = bool(re.search(r"多少|几|价格|多久|时限|上限|多大|几年|几月", question))
        type_markers: tuple[str, ...] = ()
        for pattern, markers in _ANSWER_TYPE_RULES:
            if re.search(pattern, question):
                type_markers = markers
                break

        # 知识域外检测：问题锚点词若大都未在语料中出现，则应拒答而不是硬凑片段。
        # 只有当 idf 词表来自真正的语料索引（规模足够大）时才启用，否则小样本
        # 上下文会把正常问题误判为域外。
        anchors = [
            t for t in q_terms
            if (len(t) == 1 and re.match(r"[\u4e00-\u9fff]", t)) or re.match(r"^[a-z0-9]", t)
        ]
        if self.corpus_idf is not None and len(self.corpus_idf) >= 50 and len(anchors) >= 4:
            unseen = [a for a in anchors if a not in corpus_idf]
            seen_weight = sum(min(2.5, idf_weight(a, corpus_idf)) for a in anchors if a in corpus_idf)
            total_weight = seen_weight + 3.0 * len(unseen)
            if total_weight > 0 and (3.0 * len(unseen)) / total_weight >= 0.5:
                return Generation(
                    answer="根据现有知识库无法回答该问题：问题涉及的内容不在当前知识库（云雀协作产品文档）范围内。",
                    refused=True,
                    model="extractive-v1",
                    provider="offline-extractive",
                    raw={"guard": "out_of_domain", "oov": sorted(unseen)[:8]},
                )

        top_raw = max((item.score for item in contexts), default=0.0) or 1.0
        context_terms = set().union(*(term_set(item.chunk.text) for item in contexts))
        # 区分性词：只取多字词（bigram / 英文词），单字区分度太差（「认」不等于「认证」）
        rare_terms = {
            t for t in q_terms if len(t) >= 2 and t in corpus_idf and corpus_idf[t] >= 1.15
        } & context_terms

        strict: list[tuple[float, int, int, str, float]] = []
        lenient: list[tuple[float, int, int, str, float]] = []
        for ctx_idx, item in enumerate(contexts):
            # 片段先验：检索/重排得分越高，其中的句子越可能是答案所在
            prior = max(0.0, item.score / top_raw) * (1.0 - 0.05 * min(4, ctx_idx))
            # 章节标题在切片内是独立一行，事实句（如列表项）往往不重复标题词，
            # 因此把「最近出现的标题」作为上下文词一起参与匹配，否则标题里的关键
            # 主题词（如「速率限制」）会和它下面的具体数字割裂开。
            # 章节词必须随标题切换而替换、不能跨章节累加，否则同一片段里靠后的
            # 段落会被前面章节的主题词带偏，产出与问题无关的答案。
            chunk_heading_terms = set(term_set(" ".join(item.chunk.meta.get("headings") or [])))
            section_terms = chunk_heading_terms
            for sent_idx, raw_sent in enumerate(sentences(item.chunk.text)):
                stripped = raw_sent.strip()
                if stripped.startswith("#") or _CONTEXT_HEADER.match(stripped):
                    section_terms = chunk_heading_terms | term_set(_clean_sentence(stripped))
                    continue
                sent = _clean_sentence(raw_sent)
                if not _is_evidence_sentence(sent, raw_sent):
                    continue
                st = term_set(sent) | section_terms
                if rare_terms and not (st & rare_terms):
                    continue
                cov = coverage(q_terms, st, corpus_idf)
                score = 0.3 * prior + 0.7 * cov
                # 句子自身命中了问题的区分性词（如问「团队版」时的「团队」）才算真正对口，
                # 仅靠章节标题沾边（同一张表里的其它行）不加分。
                if rare_terms and (rare_terms & term_set(sent)):
                    score += 0.06
                if q_numbers and any(num in sent for num in q_numbers):
                    score += 0.18
                if asks_number and re.search(r"\d", sent):
                    score += 0.10
                record = (score, ctx_idx, sent_idx, sent, cov)
                lenient.append(record)
                # 答案类型线索（问价格应有「元/费」等）不匹配时只作为备选
                if type_markers and not any(marker in sent for marker in type_markers):
                    continue
                strict.append(record)

        candidates = strict if strict else lenient
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        best_score = candidates[0][0] if candidates else 0.0
        if best_score < min_score:
            return Generation(
                answer="根据现有知识库无法回答该问题：检索到的片段中没有足够明确的依据。",
                refused=True,
                model="extractive-v1",
                provider="offline-extractive",
                raw={"guard": "insufficient_evidence", "best_score": round(best_score, 4), "n_contexts": len(contexts)},
            )
        relative_floor = max(min_score, best_score * 0.86)
        chosen: list[tuple[int, int, str, float]] = []
        for score, ctx_idx, sent_idx, sent, cov in candidates:
            if score < relative_floor:
                break
            if any(jaccard(term_set(sent), term_set(prev[2])) > 0.82 for prev in chosen):
                continue
            chosen.append((ctx_idx, sent_idx, sent, score))
            if len(chosen) >= max_sentences:
                break

        if not chosen:
            return Generation(
                answer="根据现有知识库无法回答该问题：检索到的片段中没有足够依据。",
                refused=True,
                model="extractive-v1",
                provider="offline-extractive",
                raw={"guard": "insufficient_evidence", "n_contexts": len(contexts)},
            )

        chosen.sort(key=lambda c: (c[0], c[1]))
        parts: list[str] = []
        citations: list[Citation] = []
        for ctx_idx, sent_idx, sent, _score in chosen:
            marker = ctx_idx + 1
            # 引用标记放在句号之前：既符合阅读习惯，也保证分句时标记与所属句子绑定
            sent = sent.rstrip("。！？.!?；; ")
            parts.append(f"{sent}[{marker}]。")
            chunk = contexts[ctx_idx].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    quote=truncate(sent, 80),
                )
            )
        answer = "".join(parts)
        if self.cfg.dedupe_sentences:
            answer = dedupe_sentences(answer)
        best = chosen[0][3]
        return Generation(
            answer=answer,
            citations=_dedupe_citations(citations),
            refused=False,
            model="extractive-v1",
            provider="offline-extractive",
            prompt_tokens=estimate_tokens(question) + estimate_tokens(format_contexts(contexts)),
            completion_tokens=estimate_tokens(answer),
            raw={"selected": len(chosen), "best_score": round(best, 4), "prompt": self.cfg.prompt},
        )

    # -- 引用解析 --------------------------------------------------------
    def _attach_citations(self, answer: str, contexts: Sequence[Retrieved]) -> list[Citation]:
        out: list[Citation] = []
        by_marker: dict[int, Citation] = {}
        for sent in sentences(answer):
            for marker in extract_citations(sent):
                if marker < 1 or marker > len(contexts):
                    continue
                if marker in by_marker:
                    continue
                chunk = contexts[marker - 1].chunk
                citation = Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    quote=truncate(sent, 80),
                )
                by_marker[marker] = citation
                out.append(citation)
        out.sort(key=lambda c: c.marker)
        return out


def _dedupe_citations(items: Sequence[Citation]) -> list[Citation]:
    seen: set[tuple[int, str]] = set()
    out: list[Citation] = []
    for item in items:
        key = (item.marker, item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
