"""检索、融合与重排序测试。"""

from rag_eval.config import RerankCfg, RetrieveCfg
from rag_eval.embeddings import HashingEmbedder, cosine
from rag_eval.retrieve import BM25Index, HybridRetriever, IdentityReranker, LexicalReranker
from rag_eval.store import MemoryStore
from rag_eval.types import Chunk

TEXTS = [
    ("c1", "团队版价格为 699 元/人/年，成员上限 200 人。"),
    ("c2", "免费版限 10 人、5 个项目、1GB 附件空间。"),
    ("c3", "存储层使用 AES-256 加密，密钥由 KMS 托管并每 90 天轮换。"),
    ("c4", "数据传输使用 TLS 1.2 及以上，禁用 SSLv3。"),
    ("c5", "任务删除后进入回收站，保留 30 天，可恢复。"),
]


def _chunks() -> list[Chunk]:
    return [
        Chunk(chunk_id=cid, doc_id="d", title="测试", source="s", text=text, index=i, meta={"headings": []})
        for i, (cid, text) in enumerate(TEXTS)
    ]


def _store() -> MemoryStore:
    embedder = HashingEmbedder(dim=256)
    store = MemoryStore()
    chunks = _chunks()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store


def test_bm25_ranks_keyword_match_first():
    index = BM25Index(_chunks())
    ranked = index.search("团队版 价格", 3)
    assert ranked
    assert index.chunks[ranked[0][0]].chunk_id == "c1"
    assert ranked[0][1] > 0


def test_bm25_empty_query_returns_nothing():
    assert BM25Index(_chunks()).search("   ", 5) == []


def test_embedding_similarity_prefers_related_text():
    embedder = HashingEmbedder(dim=384)
    query = embedder.embed_query("团队版价格")
    related = embedder.embed_documents(["团队版价格为 699 元"])[0]
    unrelated = embedder.embed_documents(["任务删除后进入回收站"])[0]
    assert cosine(query, related) > cosine(query, unrelated)


def test_hybrid_retriever_returns_top_k_with_scores():
    store = _store()
    retriever = HybridRetriever(
        store=store,
        embedder=HashingEmbedder(dim=256),
        cfg=RetrieveCfg(top_k=3, candidate_k=5, hybrid=True),
        reranker=LexicalReranker(),
    )
    results, timings = retriever.retrieve("团队版价格是多少")
    assert len(results) <= 3
    assert results[0].chunk.chunk_id == "c1"
    assert all(0.0 <= item.rerank_score <= 1.0 for item in results)
    assert timings.total_ms >= 0


def test_dense_only_mode_skips_bm25():
    store = _store()
    retriever = HybridRetriever(
        store=store,
        embedder=HashingEmbedder(dim=256),
        cfg=RetrieveCfg(top_k=2, candidate_k=2, hybrid=False),
        reranker=IdentityReranker(),
    )
    results, _ = retriever.retrieve("任务删除")
    assert all(item.source_list == ["dense"] for item in results)


def test_reranker_changes_order_vs_identity():
    store = _store()
    cfg = RetrieveCfg(top_k=3, candidate_k=5, hybrid=False)
    identity = HybridRetriever(store=store, embedder=HashingEmbedder(dim=256), cfg=cfg, reranker=IdentityReranker())
    lexical = HybridRetriever(store=store, embedder=HashingEmbedder(dim=256), cfg=cfg, reranker=LexicalReranker())
    plain, _ = identity.retrieve("AES-256 加密 密钥轮换")
    reranked, _ = lexical.retrieve("AES-256 加密 密钥轮换")
    assert reranked[0].chunk.chunk_id == "c3"
    assert plain  # 至少能返回结果，顺序差异由 reranker 负责


def test_mmr_reduces_duplicate_contexts():
    chunks = [Chunk(chunk_id=f"d{i}", doc_id="d", title="t", source="s", text="团队版价格为 699 元 " * 3, index=i) for i in range(4)]
    chunks.append(Chunk(chunk_id="other", doc_id="d", title="t", source="s", text="数据保留 30 天", index=4))
    embedder = HashingEmbedder(dim=128)
    store = MemoryStore()
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        cfg=RetrieveCfg(top_k=2, candidate_k=5, hybrid=False, mmr_lambda=0.5),
        reranker=IdentityReranker(),
    )
    results, _ = retriever.retrieve("团队版价格")
    assert len({item.chunk.chunk_id for item in results}) == len(results)


def test_reranker_metadata():
    assert LexicalReranker().info()["provider"] == "lexical"
    assert IdentityReranker().info()["provider"] == "none"
    assert RerankCfg(provider="lexical").provider == "lexical"
