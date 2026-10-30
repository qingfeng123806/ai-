"""测试共用夹具：临时配置、构造检索结果、临时数据库。"""

from __future__ import annotations

import uuid
from pathlib import Path

from rag_eval.config import Config
from rag_eval.types import CaseSpec, Chunk, Retrieved

_BASE = Path(__file__).resolve().parent.parent / "artifacts" / ".test-tmp"
_BASE.mkdir(parents=True, exist_ok=True)


def tempdir() -> Path:
    """工作区内的唯一临时目录。

    刻意不使用 ``tempfile.mkdtemp``：它创建目录时会带上仅授予所有者的 DACL，
    在受限令牌（如 Windows 沙箱）下，子目录创建会被拒绝；普通 ``mkdir``
    继承父目录权限，行为更稳定。
    """
    path = _BASE / f"case-{uuid.uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_config(tmp: str | Path, **overrides) -> Config:
    """在临时目录下构造配置：索引、数据库、报告都写入 tmp，不污染仓库。"""
    tmp = Path(tmp)
    data = {
        "project": {"artifacts_dir": str(tmp / "artifacts"), "db_path": str(tmp / "artifacts" / "eval.db"), "seed": 42},
        "ingest": {"paths": ["data/corpus"], "chunker": "recursive", "chunk_size": 420},
        "embedding": {"provider": "hashing", "dim": 128},
        "store": {"provider": "memory", "persist_dir": str(tmp / "artifacts" / "index")},
        "retrieve": {"top_k": 4, "candidate_k": 8, "hybrid": True},
        "rerank": {"provider": "lexical", "top_n": 4},
        "generate": {"provider": "offline", "prompt": "grounded_cite_v1"},
        "evaluate": {"bootstrap_samples": 200, "metrics": [
            "hit_rate", "precision_at_k", "context_recall", "faithfulness",
            "answer_relevancy", "answer_completeness", "hallucination_rate",
            "citation_accuracy", "citation_coverage", "refusal_appropriateness",
            "judge_overall", "latency_p50",
        ]},
        "report": {"out_dir": str(tmp / "artifacts" / "reports")},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config.from_dict(data)


def make_chunk(text: str, chunk_id: str = "c1", title: str = "测试文档", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        title=title,
        source="data/corpus/test.md",
        text=text,
        index=index,
        meta={"headings": [title]},
    )


def make_retrieved(text: str, chunk_id: str = "c1", score: float = 0.8, rank: int = 1) -> Retrieved:
    return Retrieved(chunk=make_chunk(text, chunk_id=chunk_id), rerank_score=score, rank=rank)


def make_case(question: str = "团队版价格是多少？", **kwargs) -> CaseSpec:
    payload = {
        "case_id": "t-001",
        "split": "core",
        "question": question,
        "reference_answer": "团队版价格为 699 元/人/年。",
        "reference_contexts": ["团队版 | 699 元/人/年"],
        "expected_keywords": ["699", "团队版"],
    }
    payload.update(kwargs)
    return CaseSpec.from_dict(payload)
