"""RAG 流水线编排：建索引（加载→切片→向量化→入库）与在线问答（检索→重排→生成）。

索引按「语料内容 + 切片参数 + 向量模型」的内容指纹分目录缓存，
所以 A/B 实验切换切片策略时会自动建独立索引，互不污染，重复运行直接命中缓存。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import Config, EMBEDDERS, STORES
from .embeddings import CachedEmbedder, build_embedder
from .generate import Answerer, format_contexts
from .ingest import ChunkStats, chunk_documents, corpus_signature, ingest_stats, load_corpus
from .llm import build_llm, has_api_key
from .retrieve import HybridRetriever, RetrievalTimings, build_reranker
from .store import VectorStore, build_store
from .text import build_idf, short_hash
from .types import Chunk, Document, Generation, LatencyBreakdown, Retrieved

INDEX_META = "index.meta.json"


@dataclass
class KnowledgeIndex:
    docs: list[Document]
    chunks: list[Chunk]
    store: VectorStore
    embedder: CachedEmbedder
    signature: str
    stats: ChunkStats
    cached: bool = False
    built_ms: float = 0.0
    path: str = ""

    def info(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "documents": len(self.docs),
            "chunks": len(self.chunks),
            "cached": self.cached,
            "built_ms": round(self.built_ms, 2),
            "path": self.path,
            "chunk_stats": self.stats.to_dict(),
            "embedding": self.embedder.info(),
            "store": self.store.stats(),
        }


@dataclass
class QueryResult:
    question: str
    contexts: list[Retrieved]
    generation: Generation
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    timings: RetrievalTimings = field(default_factory=RetrievalTimings)
    prompt: str = ""
    context_block: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.generation.answer,
            "refused": self.generation.refused,
            "citations": [c.to_dict() for c in self.generation.citations],
            "contexts": [c.to_dict() for c in self.contexts],
            "latency": self.latency.to_dict(),
            "timings": self.timings.to_dict(),
            "model": self.generation.model,
            "provider": self.generation.provider,
        }


def _embedder_signature(cfg: Config, embedder: CachedEmbedder) -> str:
    return f"{embedder.kind}:{embedder.name}:{embedder.dim}:{cfg.embedding.normalize}:{cfg.embedding.query_prefix}"


def build_index(cfg: Config, force: bool = False, verbose: bool = False) -> KnowledgeIndex:
    started = time.perf_counter()
    docs = load_corpus(cfg.ingest.paths, cfg.ingest.include_globs)
    if not docs:
        raise RuntimeError("语料为空：请检查 ingest.paths 下的文档")

    artifacts = Path(cfg.project.artifacts_dir)
    cache_dir = artifacts / ".cache"
    embedder = build_embedder(cfg.embedding, cache_dir=cache_dir)
    signature = short_hash(corpus_signature(docs, cfg.ingest) + "|" + _embedder_signature(cfg, embedder), 16)

    index_dir = Path(cfg.store.persist_dir) / signature
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / INDEX_META

    store = build_store(cfg.store, embedder.dim, persist_dir=index_dir)
    cached = False
    chunks: list[Chunk] = []
    if not force and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("signature") == signature and store.load():
            chunks = store.all_chunks()
            cached = len(chunks) > 0
    if not cached:
        store.reset() if hasattr(store, "reset") else None
        chunks = chunk_documents(docs, cfg.ingest)
        if verbose:
            print(f"  · 切片完成：{len(docs)} 篇文档 → {len(chunks)} 个切片")
        vectors: list[list[float]] = []
        batch = max(4, cfg.embedding.batch_size)
        for i in range(0, len(chunks), batch):
            batch_chunks = chunks[i : i + batch]
            vectors.extend(embedder.embed_documents([c.text for c in batch_chunks]))
            if verbose and (i // batch) % 5 == 0:
                print(f"  · 向量化 {min(i + batch, len(chunks))}/{len(chunks)}", end="\r")
        store.add(chunks, vectors)
        store.persist()
        meta_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "chunks": len(chunks),
                    "dim": embedder.dim,
                    "embedder": embedder.info(),
                    "store": store.stats(),
                    "chunker": cfg.ingest.chunker,
                    "documents": [d.to_dict() for d in docs],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if verbose:
            print()
    stats = ingest_stats(docs, chunks, cfg.ingest.chunker)
    elapsed = (time.perf_counter() - started) * 1000
    return KnowledgeIndex(
        docs=docs,
        chunks=chunks,
        store=store,
        embedder=embedder,
        signature=signature,
        stats=stats,
        cached=cached,
        built_ms=elapsed,
        path=str(index_dir),
    )


class RAGPipeline:
    """把检索器、重排器与生成器组装为可复用的问答流水线。"""

    def __init__(self, cfg: Config, index: KnowledgeIndex):
        self.cfg = cfg
        self.index = index
        self.retriever = HybridRetriever(
            store=index.store,
            embedder=index.embedder,
            cfg=cfg.retrieve,
            reranker=build_reranker(cfg.rerank),
        )
        llm = build_llm(cfg.generate)
        self.answerer = Answerer(
            cfg.generate,
            llm,
            corpus_idf=build_idf([c.text for c in index.chunks]),
        )
        self._llm_fallback = cfg.generate.provider not in ("offline",) and llm is None

    # -- 元信息 ----------------------------------------------------------
    @property
    def mode(self) -> str:
        live_llm = self.answerer.llm is not None
        live_embed = getattr(self.index.embedder, "kind", "hashing") != "hashing"
        if live_llm and live_embed:
            return "live"
        if live_llm or live_embed:
            return "hybrid"
        return "offline"

    def info(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "index": self.index.info(),
            "retrieval": self.retriever.info(),
            "generation": self.answerer.info(),
            "llm_fallback": self._llm_fallback,
            "config_fingerprint": self.cfg.fingerprint(),
        }

    def providers(self) -> dict[str, str]:
        gem = self.index.embedder.info()
        store = self.index.store.stats()
        gen = self.answerer.info()
        return {
            "embedding": f"{gem.get('provider')}:{gem.get('name')}",
            "store": f"{store.get('provider')}:{store.get('name') or store.get('provider')}",
            "llm": f"{gen.get('provider')}:{gen.get('model')}",
            "prompt": str(self.cfg.generate.prompt),
            "reranker": str(self.retriever.reranker.name),
        }

    # -- 问答 ------------------------------------------------------------
    def query(self, question: str) -> QueryResult:
        t0 = time.perf_counter()
        contexts, timings = self.retriever.retrieve(question)
        t1 = time.perf_counter()
        generation = self.answerer.answer(question, contexts)
        t2 = time.perf_counter()
        total_ms = (t2 - t0) * 1000
        latency = LatencyBreakdown(
            retrieval_ms=timings.dense_ms + timings.lexical_ms + timings.fuse_ms,
            rerank_ms=timings.rerank_ms,
            generation_ms=(t2 - t1) * 1000,
            total_ms=total_ms,
        )
        return QueryResult(
            question=question,
            contexts=contexts,
            generation=generation,
            latency=latency,
            timings=timings,
            prompt=str(self.cfg.generate.prompt),
            context_block=format_contexts(contexts, self.cfg.generate.max_context_chars),
        )


def load_pipeline(cfg: Config, force: bool = False, verbose: bool = False) -> RAGPipeline:
    index = build_index(cfg, force=force, verbose=verbose)
    return RAGPipeline(cfg, index)


def describe_environment(cfg: Config) -> dict[str, Any]:
    """报告各子系统的实际可用后端，便于看板与日志判断运行模式。"""
    from .embeddings import BgeEmbedder
    from .store import chroma_available, milvus_available

    bge_ok = False
    try:
        BgeEmbedder(model_name=cfg.embedding.model).embed_query("probe")
        bge_ok = True
    except Exception:
        bge_ok = False
    return {
        "embedding": {
            "configured": cfg.embedding.provider,
            "bge_available": bge_ok,
            "fallback": "hashing" if not bge_ok else None,
        },
        "store": {
            "configured": cfg.store.provider,
            "chroma_available": chroma_available(),
            "milvus_available": milvus_available(),
        },
        "llm": {
            "configured": cfg.generate.provider,
            "api_key_present": has_api_key(cfg.generate),
            "fallback": None if has_api_key(cfg.generate) else "offline-extractive",
        },
    }
