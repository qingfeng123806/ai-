"""检索层：BM25 稀疏检索、稠密检索、RRF 融合与重排序。

设计要点
--------
1. 稠密与稀疏各自召回 ``candidate_k`` 条，用 RRF（Reciprocal Rank Fusion）融合，
   避免两种分数量纲不一致（余弦 0~1 vs BM25 无上界）。
2. ``rerank`` 是独立的可插拔阶段：BGE 交叉编码器缺失时降级为词法重排器，
   它同时利用查询词覆盖、短语命中与章节标题匹配，仍能显著改善排序。
3. 每个阶段都返回耗时，便于在看板里区分「检索慢」还是「生成慢」。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from .config import RerankCfg, RetrieveCfg
from .embeddings import Embedder
from .store import VectorStore
from .text import coverage, jaccard, normalize_ws, term_set, terms
from .types import Chunk, Retrieved


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25，倒排索引实现，适合几千到十万级切片的内存检索。"""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = list(chunks)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        for idx, chunk in enumerate(self.chunks):
            tokens = terms(chunk.text)
            self.doc_len.append(max(1, len(tokens)))
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, count in tf.items():
                self.postings.setdefault(tok, []).append((idx, count))
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0
        self.n_docs = len(self.chunks)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.postings:
            return []
        q_terms = terms(query)
        if not q_terms:
            return []
        scores: dict[int, float] = {}
        for term in set(q_terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (self.k1 + 1) / (denom or 1.0)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def stats(self) -> dict[str, Any]:
        return {
            "docs": self.n_docs,
            "vocab": len(self.postings),
            "avgdl": round(self.avgdl, 1),
        }


# --------------------------------------------------------------------------
# 重排序
# --------------------------------------------------------------------------

class Reranker(Protocol):
    name: str
    kind: str

    def rerank(self, query: str, candidates: Sequence[Retrieved], top_n: int) -> list[Retrieved]: ...

    def info(self) -> dict[str, Any]: ...


class IdentityReranker:
    """不重排，仅截断。"""

    kind = "none"

    def __init__(self) -> None:
        self.name = "no-rerank"

    def rerank(self, query: str, candidates: Sequence[Retrieved], top_n: int) -> list[Retrieved]:
        out = list(candidates[:top_n])
        for i, item in enumerate(out):
            item.rank = i + 1
        return out

    def info(self) -> dict[str, Any]:
        return {"provider": "none", "name": self.name}


_PHRASE_SPLIT = re.compile(r"[。！？；,，.?!;\n]+")


class LexicalReranker:
    """词法重排器：查询词加权覆盖 + 短语命中 + 标题匹配 + 稠密分先验。"""

    kind = "lexical"

    def __init__(self, lexical_weight: float = 0.55, dense_weight: float = 0.25, title_weight: float = 0.2):
        self.name = "lexical-overlap"
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.title_weight = title_weight

    def _score(self, query: str, item: Retrieved, q_terms: set[str]) -> float:
        text = item.chunk.text
        text_terms = term_set(text)
        cover = coverage(q_terms, text_terms)
        # 整句短语命中：把查询按标点切成短语，命中则额外加分
        phrases = [p.strip() for p in _PHRASE_SPLIT.split(query) if len(p.strip()) >= 4]
        phrase_hits = sum(1 for p in phrases if p in text)
        phrase_bonus = min(1.0, phrase_hits / max(1, len(phrases))) if phrases else 0.0
        # 章节标题命中
        headings = " ".join(item.chunk.meta.get("headings") or [])
        title_hit = 1.0 if q_terms & term_set(headings) else 0.0
        dense = max(0.0, min(1.0, item.dense_score))
        return (
            self.lexical_weight * (0.7 * cover + 0.3 * phrase_bonus)
            + self.dense_weight * dense
            + self.title_weight * title_hit
        )

    def rerank(self, query: str, candidates: Sequence[Retrieved], top_n: int) -> list[Retrieved]:
        q_terms = term_set(query)
        scored: list[Retrieved] = []
        for item in candidates:
            item.rerank_score = round(self._score(query, item, q_terms), 6)
            scored.append(item)
        scored.sort(key=lambda it: (-(it.rerank_score or 0.0), it.chunk.chunk_id))
        out = scored[:top_n]
        for i, item in enumerate(out):
            item.rank = i + 1
        return out

    def info(self) -> dict[str, Any]:
        return {"provider": "lexical", "name": self.name}


class BgeReranker:
    """BGE 交叉编码器重排序（sentence-transformers CrossEncoder / FlagReranker）。"""

    kind = "bge"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 16):
        self.name = f"bge-reranker:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None
        self._backend = ""

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(self.model_name)
            self._backend = "sentence-transformers"
            return self._model
        except Exception:
            pass
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            self._model = FlagReranker(self.model_name, use_fp16=False)
            self._backend = "FlagEmbedding"
            return self._model
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"无法加载 BGE 重排序模型 {self.model_name}：{exc}") from exc

    def rerank(self, query: str, candidates: Sequence[Retrieved], top_n: int) -> list[Retrieved]:
        if not candidates:
            return []
        try:
            model = self._load()
        except Exception:
            # 运行期模型不可用时不能让整次评测失败：退回词法重排
            self._model = False
            return LexicalReranker().rerank(query, candidates, top_n)
        if self._model is False:
            return LexicalReranker().rerank(query, candidates, top_n)
        pairs = [(query, item.chunk.text) for item in candidates]
        if self._backend == "sentence-transformers":
            raw = model.predict(pairs, batch_size=self.batch_size)
        else:  # pragma: no cover
            raw = [model.compute_score(pair) for pair in pairs]
        for item, score in zip(candidates, raw):
            value = float(score)
            # 交叉编码器输出为 logits，用 sigmoid 映射到 0~1 便于跨实验比较
            item.rerank_score = round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))), 6)
        ordered = sorted(candidates, key=lambda it: (-(it.rerank_score or 0.0), it.chunk.chunk_id))
        out = ordered[:top_n]
        for i, item in enumerate(out):
            item.rank = i + 1
        return out

    def info(self) -> dict[str, Any]:
        return {"provider": "bge", "name": self.name, "backend": self._backend or "unloaded"}


def build_reranker(cfg: RerankCfg) -> Reranker:
    provider = (cfg.provider or "auto").lower()
    if provider == "none":
        return IdentityReranker()
    if provider == "lexical":
        return LexicalReranker()
    candidate = BgeReranker(cfg.model, cfg.batch_size)
    if provider == "bge":
        return candidate
    try:  # auto：先探活，避免评测中途才发现模型缺失
        candidate._load()
        return candidate
    except Exception:
        return LexicalReranker()


# --------------------------------------------------------------------------
# 混合检索
# --------------------------------------------------------------------------

@dataclass
class RetrievalTimings:
    dense_ms: float = 0.0
    lexical_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.dense_ms + self.lexical_ms + self.fuse_ms + self.rerank_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "dense_ms": round(self.dense_ms, 2),
            "lexical_ms": round(self.lexical_ms, 2),
            "fuse_ms": round(self.fuse_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
        }


@dataclass
class HybridRetriever:
    store: VectorStore
    embedder: Embedder
    cfg: RetrieveCfg
    reranker: Reranker
    bm25: BM25Index | None = None
    _chunk_map: dict[str, Chunk] = field(default_factory=dict)
    _loaded: bool = False

    def __post_init__(self) -> None:
        if not self._loaded:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        chunks = self.store.all_chunks()
        self._chunk_map = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks) if chunks else None
        self._loaded = True

    # -- 融合 ------------------------------------------------------------
    def _rrf(self, dense: Sequence[tuple[Chunk, float]], lexical: Sequence[tuple[int, float]]) -> list[Retrieved]:
        k = max(1, self.cfg.rrf_k)
        items: dict[str, Retrieved] = {}
        for rank, (chunk, score) in enumerate(dense, start=1):
            item = items.setdefault(chunk.chunk_id, Retrieved(chunk=chunk))
            item.dense_score = float(score)
            item.source_list.append("dense")
            item.fusion_score += self.cfg.dense_weight / (k + rank)
        if self.bm25 is not None:
            for rank, (doc_idx, score) in enumerate(lexical, start=1):
                chunk = self.bm25.chunks[doc_idx]
                item = items.setdefault(chunk.chunk_id, Retrieved(chunk=chunk))
                item.lexical_score = float(score)
                item.source_list.append("bm25")
                item.fusion_score += self.cfg.lexical_weight / (k + rank)
        fused = sorted(items.values(), key=lambda it: (-it.fusion_score, it.chunk.chunk_id))
        top = fused[0].fusion_score if fused else 1.0
        for item in fused:
            item.fusion_score = item.fusion_score / (top or 1.0)
        return fused

    def _mmr(self, items: list[Retrieved], lambda_: float) -> list[Retrieved]:
        """基于词面相似度的 MMR，抑制同质片段挤占上下文。"""
        if lambda_ <= 0 or len(items) <= 2:
            return items
        selected: list[Retrieved] = [items[0]]
        pool = items[1:]
        while pool:
            best_idx = 0
            best_val = -1e9
            sel_terms = [term_set(s.chunk.text) for s in selected]
            for i, cand in enumerate(pool):
                cand_terms = term_set(cand.chunk.text)
                redundancy = max((jaccard(cand_terms, st) for st in sel_terms), default=0.0)
                value = lambda_ * cand.fusion_score - (1 - lambda_) * redundancy
                if value > best_val:
                    best_val = value
                    best_idx = i
            selected.append(pool.pop(best_idx))
        return selected

    # -- 主流程 ----------------------------------------------------------
    def retrieve(self, query: str) -> tuple[list[Retrieved], RetrievalTimings]:
        timings = RetrievalTimings()
        query = normalize_ws(query)
        if not query:
            return [], timings
        candidate_k = max(self.cfg.candidate_k, self.cfg.top_k)

        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        dense_pairs = self.store.search(qvec, candidate_k)
        timings.dense_ms = (time.perf_counter() - t0) * 1000

        lexical_pairs: list[tuple[int, float]] = []
        if self.cfg.hybrid and self.bm25 is not None:
            t0 = time.perf_counter()
            lexical_pairs = self.bm25.search(query, candidate_k)
            timings.lexical_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if self.cfg.hybrid:
            fused = self._rrf(dense_pairs, lexical_pairs)
        else:
            fused = [
                Retrieved(
                    chunk=chunk,
                    dense_score=float(score),
                    fusion_score=max(0.0, float(score)),
                    source_list=["dense"],
                )
                for chunk, score in dense_pairs
            ]
        if self.cfg.mmr_lambda > 0:
            fused = self._mmr(fused, self.cfg.mmr_lambda)
        if self.cfg.min_score > 0:
            filtered = [it for it in fused if it.fusion_score >= self.cfg.min_score]
            fused = filtered or fused[: self.cfg.top_k]
        timings.fuse_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        chosen = self.reranker.rerank(query, fused[:candidate_k], self.cfg.top_k)
        timings.rerank_ms = (time.perf_counter() - t0) * 1000
        return chosen, timings

    def info(self) -> dict[str, Any]:
        return {
            "retriever": "hybrid-rrf" if self.cfg.hybrid else "dense",
            "top_k": self.cfg.top_k,
            "candidate_k": self.cfg.candidate_k,
            "rrf_k": self.cfg.rrf_k,
            "bm25": self.bm25.stats() if self.bm25 else None,
            "reranker": self.reranker.info(),
        }
