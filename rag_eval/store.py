"""向量库适配层：内存暴力检索 / Chroma / Milvus。

三者暴露同一接口，评测与 A/B 实验只依赖抽象，切换向量库不需要改评测代码。
``auto`` 会优先使用已安装的 Chroma（持久化），否则退回内存索引。
"""

from __future__ import annotations

import array
import json
import math
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import StoreCfg
from .types import Chunk

CHUNK_FILE = "chunks.jsonl"
VECTOR_FILE = "vectors.f32"
META_FILE = "index.meta.json"


class VectorStore(ABC):
    kind = "base"
    name = "base"

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def count(self) -> int: ...

    def persist(self) -> None:
        return None

    def load(self) -> bool:
        return False

    def reset(self) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"provider": self.kind, "chunks": self.count()}

    def all_chunks(self) -> list[Chunk]:
        return []


class MemoryStore(VectorStore):
    """纯标准库实现：float32 二进制向量 + JSONL 切片，余弦相似度暴力检索。"""

    kind = "memory"

    def __init__(self, persist_dir: str | Path | None = None, normalize: bool = True):
        self.dir = Path(persist_dir) if persist_dir else None
        self.normalize = normalize
        self.name = "memory-cosine"
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._index: dict[str, int] = {}

    # -- 写入 ------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"切片数 {len(chunks)} 与向量数 {len(vectors)} 不一致")
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in self._index:
                pos = self._index[chunk.chunk_id]
                self._chunks[pos] = chunk
                self._vectors[pos] = list(vector)
                continue
            self._index[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            self._vectors.append(list(vector))

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        if not self._vectors or k <= 0:
            return []
        qn = math.sqrt(sum(v * v for v in vector)) or 1.0
        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(self._vectors):
            if self.normalize:
                score = sum(a * b for a, b in zip(vector, vec)) / qn
            else:
                vn = math.sqrt(sum(v * v for v in vec)) or 1.0
                score = sum(a * b for a, b in zip(vector, vec)) / (qn * vn)
            scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self._chunks[i], score) for score, i in scored[:k]]

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    # -- 持久化 ----------------------------------------------------------
    def persist(self) -> None:
        if not self.dir:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / CHUNK_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        flat = array.array("f")
        for vec in self._vectors:
            flat.extend(vec)
        with (self.dir / VECTOR_FILE).open("wb") as fh:
            flat.tofile(fh)
        (self.dir / META_FILE).write_text(
            json.dumps(
                {
                    "provider": self.kind,
                    "name": self.name,
                    "chunks": len(self._chunks),
                    "dim": len(self._vectors[0]) if self._vectors else 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.dir:
            return False
        chunk_path = self.dir / CHUNK_FILE
        vec_path = self.dir / VECTOR_FILE
        if not chunk_path.exists() or not vec_path.exists():
            return False
        meta_path = self.dir / META_FILE
        dim = 0
        if meta_path.exists():
            try:
                dim = int(json.loads(meta_path.read_text(encoding="utf-8")).get("dim") or 0)
            except Exception:
                dim = 0
        chunks: list[Chunk] = []
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append(Chunk.from_dict(json.loads(line)))
        if not chunks:
            return False
        if dim <= 0:
            return False
        flat = array.array("f")
        with vec_path.open("rb") as fh:
            flat.fromfile(fh, len(chunks) * dim)
        if len(flat) != len(chunks) * dim:
            return False
        self._chunks = chunks
        self._vectors = [list(flat[i * dim : (i + 1) * dim]) for i in range(len(chunks))]
        self._index = {c.chunk_id: i for i, c in enumerate(chunks)}
        return True

    def reset(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._index.clear()
        if self.dir and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.kind,
            "name": self.name,
            "chunks": len(self._chunks),
            "dim": len(self._vectors[0]) if self._vectors else 0,
            "persist_dir": str(self.dir) if self.dir else None,
        }


class ChromaStore(VectorStore):
    """Chroma 持久化向量库适配器（需安装 chromadb）。"""

    kind = "chroma"

    def __init__(self, persist_dir: str | Path, collection: str, metric: str = "cosine"):
        import chromadb  # type: ignore
        from chromadb.config import Settings  # type: ignore

        self.name = f"chroma:{collection}"
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._space = "cosine" if metric == "cosine" else ("l2" if metric == "l2" else "ip")
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": self._space}
        )

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = []
        for chunk in chunks:
            metas.append(
                {
                    "doc_id": chunk.doc_id[:200],
                    "title": chunk.title[:200],
                    "source": chunk.source[:400],
                    "index": int(chunk.index),
                    "payload": json.dumps(chunk.to_dict(), ensure_ascii=False),
                }
            )
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=[list(v) for v in vectors])

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.query(query_embeddings=[list(vector)], n_results=max(1, k))
        out: list[tuple[Chunk, float]] = []
        ids = (result.get("ids") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, _cid in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            try:
                chunk = Chunk.from_dict(json.loads(meta.get("payload") or "{}"))
            except Exception:
                chunk = Chunk(chunk_id=ids[i], doc_id="", title="", source="", text="")
            distance = float(dists[i]) if i < len(dists) else 1.0
            score = 1.0 - distance if self._space == "cosine" else 1.0 / (1.0 + distance)
            out.append((chunk, score))
        return out

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    def all_chunks(self) -> list[Chunk]:
        try:
            data = self._collection.get(include=["metadatas"])
        except Exception:
            return []
        out: list[Chunk] = []
        for meta in data.get("metadatas") or []:
            try:
                out.append(Chunk.from_dict(json.loads((meta or {}).get("payload") or "{}")))
            except Exception:
                continue
        return out

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection.name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name, metadata={"hnsw:space": self._space}
        )

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.kind,
            "name": self.name,
            "chunks": self.count(),
            "persist_dir": str(self.dir),
            "space": self._space,
        }


class MilvusStore(VectorStore):
    """Milvus 适配器（需 pymilvus 与可用的 Milvus 服务）。"""

    kind = "milvus"

    def __init__(self, uri: str, collection: str, dim: int, metric: str = "COSINE"):
        from pymilvus import (  # type: ignore
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            utility,
        )

        self.name = f"milvus:{collection}"
        self._metric = metric.upper()
        connections.connect(alias="default", uri=uri)
        if not utility.has_collection(collection):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, description="kb rag eval chunks")
            coll = Collection(collection, schema=schema)
            coll.create_index(
                field_name="vector",
                index_params={"index_type": "AUTOINDEX", "metric_type": self._metric, "params": {}},
            )
        self._collection = Collection(collection)
        self._collection.load()

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        payloads = [json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks]
        self._collection.insert(
            [
                [c.chunk_id for c in chunks],
                [list(v) for v in vectors],
                payloads,
            ]
        )
        self._collection.flush()

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        result = self._collection.search(
            data=[list(vector)], anns_field="vector", param={"metric_type": self._metric},
            limit=max(1, k), output_fields=["payload"],
        )
        out: list[tuple[Chunk, float]] = []
        for hit in (result[0] if result else []):
            payload = hit.entity.get("payload")
            try:
                chunk = Chunk.from_dict(json.loads(payload))
            except Exception:
                chunk = Chunk(chunk_id=str(hit.id), doc_id="", title="", source="", text="")
            distance = float(hit.distance)
            score = distance if self._metric == "COSINE" else 1.0 / (1.0 + distance)
            out.append((chunk, score))
        return out

    def count(self) -> int:
        try:
            self._collection.flush()
            return int(self._collection.num_entities)
        except Exception:
            return 0

    def stats(self) -> dict[str, Any]:
        return {"provider": self.kind, "name": self.name, "chunks": self.count(), "metric": self._metric}


def build_store(cfg: StoreCfg, dim: int, persist_dir: str | Path | None = None) -> VectorStore:
    """按配置构造向量库，``auto`` 优先 Chroma。"""
    provider = (cfg.provider or "auto").lower()
    target = persist_dir or cfg.persist_dir
    if provider in ("auto", "chroma"):
        try:
            return ChromaStore(target, cfg.collection, cfg.metric)
        except Exception:
            if provider == "chroma":
                raise
    if provider == "milvus":
        return MilvusStore(cfg.milvus_uri, cfg.collection, dim=dim, metric="COSINE")
    return MemoryStore(target, normalize=cfg.metric == "cosine")


def chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401

        return True
    except Exception:
        return False


def milvus_available() -> bool:
    try:
        import pymilvus  # noqa: F401

        return True
    except Exception:
        return False
