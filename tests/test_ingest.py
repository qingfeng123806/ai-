"""加载与切片测试。"""

from rag_eval.config import IngestCfg
from rag_eval.ingest import (
    LoaderError,
    chunk_documents,
    chunk_fixed,
    chunk_heading,
    chunk_recursive,
    corpus_signature,
    ingest_stats,
    load_corpus,
    load_document,
)
from rag_eval.types import Document

SAMPLE = """# 套餐说明

## 团队版

团队版价格为 699 元/人/年，成员上限 200 人。支持自动化规则与工时统计。

## 专业版

专业版价格为 1299 元/人/年，支持 SSO、审计日志与沙箱环境。
""" + "补充说明：" + "这是一段用于测试切片长度的填充文本。" * 20


def _doc() -> Document:
    return Document(doc_id="d1", title="套餐说明", source="data/corpus/x.md", text=SAMPLE)


def test_load_corpus_reads_all_documents():
    docs = load_corpus(["data/corpus"])
    assert len(docs) >= 6
    assert all(doc.text for doc in docs)
    assert any("价格" in doc.text for doc in docs)


def test_load_document_missing_path_raises():
    try:
        load_corpus(["data/does-not-exist"])
    except LoaderError as exc:
        assert "不存在" in str(exc)
    else:
        raise AssertionError("不存在的路径应当报错")


def test_chunkers_respect_size_bounds():
    cfg = IngestCfg(chunk_size=300, chunk_overlap=40, min_chunk_chars=20)
    for fn in (chunk_fixed, chunk_recursive, chunk_heading):
        chunks = fn(_doc(), cfg)
        assert chunks, fn.__name__
        assert all(len(c.text) <= cfg.chunk_size + 80 for c in chunks), fn.__name__
        assert [c.index for c in chunks] == list(range(len(chunks)))


def test_heading_chunker_keeps_section_path():
    cfg = IngestCfg(chunker="heading", chunk_size=400, chunk_overlap=0, heading_aware=True)
    chunks = chunk_heading(_doc(), cfg)
    assert any("【" in c.text for c in chunks)
    assert any(c.meta.get("headings") for c in chunks)


def test_chunk_ids_are_unique_and_stable():
    cfg = IngestCfg(chunker="recursive", chunk_size=200, chunk_overlap=20)
    first = chunk_documents([_doc()], cfg)
    second = chunk_documents([_doc()], cfg)
    assert len({c.chunk_id for c in first}) == len(first)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_corpus_signature_changes_with_params():
    cfg = IngestCfg(chunker="recursive", chunk_size=400)
    other = IngestCfg(chunker="recursive", chunk_size=200)
    doc = _doc()
    assert corpus_signature([doc], cfg) != corpus_signature([doc], other)


def test_ingest_stats():
    cfg = IngestCfg(chunker="fixed", chunk_size=200, chunk_overlap=20)
    chunks = chunk_documents([_doc()], cfg)
    stats = ingest_stats([_doc()], chunks, "fixed")
    assert stats.documents == 1
    assert stats.chunks == len(chunks)
    assert stats.max_chars >= stats.min_chars
    assert stats.to_dict()["chunker"] == "fixed"


def test_chunk_text_covers_corpus_content():
    cfg = IngestCfg(chunker="fixed", chunk_size=150, chunk_overlap=30, min_chunk_chars=10)
    chunks = chunk_documents([_doc()], cfg)
    joined = "".join(c.text for c in chunks)
    assert "699" in joined and "1299" in joined
