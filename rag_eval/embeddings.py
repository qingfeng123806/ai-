"""向量化层：BGE 优先，缺失时降级为确定性哈希向量化器。

- ``BgeEmbedder``：接入 sentence-transformers / FlagEmbedding 的真实 BGE 模型，
  支持检索指令前缀（BGE 中文模型推荐 ``为这个句子生成表示以用于检索相关文章：``）。
- ``HashingEmbedder``：字符 2-3 gram 带符号哈希 + 次线性 TF + L2 归一化。
  它不依赖任何权重文件，检索行为稳定可复现，是中英文混排语料上可用的
  通用基线，也是离线/CI 环境下的默认后端。
- ``CachedEmbedder``：进程内 + 磁盘缓存，避免实验矩阵重复编码同一批切片。
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .config import EmbeddingCfg
from .text import char_ngrams, normalize_ws


class EmbeddingUnavailable(RuntimeError):
    """所需的向量化后端不可用（模型缺失 / 依赖未安装）。"""


class Embedder(Protocol):
    name: str
    kind: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    inv = 1.0 / norm
    return [v * inv for v in vector]


@dataclass
class HashingEmbedder:
    """无需训练权重的确定性向量化器（字符 n-gram 特征哈希）。"""

    dim: int = 384
    low: int = 2
    high: int = 3
    normalize: bool = True
    query_prefix: str = ""
    name: str = "hashing-char2_3"
    kind: str = "hashing"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        grams = char_ngrams(normalize_ws(text), self.low, self.high)
        if not grams:
            return vec
        counts: dict[int, float] = {}
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dim
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            counts[index] = counts.get(index, 0.0) + sign
        for index, value in counts.items():
            # 次线性 TF，抑制长文档中高频 n-gram 的支配作用
            vec[index] = math.copysign(1.0 + math.log1p(abs(value)), value)
        return _l2_normalize(vec) if self.normalize else vec

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector((self.query_prefix or "") + text)

    def info(self) -> dict[str, Any]:
        return {"provider": "hashing", "name": self.name, "dim": self.dim, "mode": "offline"}


@dataclass
class BgeEmbedder:
    """BGE 稠密向量模型（sentence-transformers 或 FlagEmbedding 后端）。"""

    model_name: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 384
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = ""
    doc_prefix: str = ""
    device: str = ""
    name: str = ""
    kind: str = "bge"

    def __post_init__(self) -> None:
        self.name = self.name or f"bge:{self.model_name}"
        self._model: Any = None
        self._backend = ""

    # -- 加载 ------------------------------------------------------------
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:  # 优先 sentence-transformers
            from sentence_transformers import SentenceTransformer  # type: ignore

            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
            self._backend = "sentence-transformers"
            self.dim = int(self._model.get_sentence_embedding_dimension() or self.dim)
            return self._model
        except Exception:  # pragma: no cover - 依赖可选
            pass
        try:  # 其次 FlagEmbedding
            from FlagEmbedding import FlagModel  # type: ignore

            self._model = FlagModel(self.model_name, use_fp16=False)
            self._backend = "FlagEmbedding"
            return self._model
        except Exception as exc:  # pragma: no cover
            raise EmbeddingUnavailable(
                f"无法加载 BGE 模型 {self.model_name}：{exc}. "
                "请安装 sentence-transformers 或 FlagEmbedding，并确保模型权重可访问。"
            ) from exc

    # -- 编码 ------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        if self._backend == "sentence-transformers":
            vectors = model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return [list(map(float, v)) for v in vectors]
        raw = model.encode(list(texts), batch_size=self.batch_size)  # pragma: no cover
        out = []
        for v in raw:
            vec = [float(x) for x in v]
            out.append(_l2_normalize(vec) if self.normalize else vec)
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [(self.doc_prefix or "") + t for t in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([(self.query_prefix or "") + text])[0]

    def info(self) -> dict[str, Any]:
        return {
            "provider": "bge",
            "name": self.name,
            "dim": self.dim,
            "mode": "live",
            "backend": self._backend or "unloaded",
        }


class CachedEmbedder:
    """给任意 embedder 加进程内 / 磁盘缓存。"""

    def __init__(self, inner: Embedder, cache_dir: str | Path | None = None, enabled: bool = True):
        self.inner = inner
        self.name = getattr(inner, "name", "embedder")
        self.kind = getattr(inner, "kind", "unknown")
        self.dim = int(getattr(inner, "dim", 384))
        self.enabled = enabled
        self._docs: dict[str, list[float]] = {}
        self._queries: dict[str, list[float]] = {}
        self._dir = Path(cache_dir) if cache_dir and enabled else None
        self.hits = 0
        self.misses = 0
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    @property
    def _disk_path(self) -> Path | None:
        if not self._dir:
            return None
        return self._dir / f"emb_{self.kind}_{self.dim}.pkl"

    def _load_disk(self) -> None:
        path = self._disk_path
        if not path or not path.exists():
            return
        try:
            data = pickle.loads(path.read_bytes())
        except Exception:
            return
        if isinstance(data, dict):
            self._docs.update(data.get("docs", {}))

    def _save_disk(self) -> None:
        path = self._disk_path
        if not path:
            return
        try:
            path.write_bytes(pickle.dumps({"docs": self._docs}, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            pass

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.enabled:
            return self.inner.embed_documents(texts)
        if self._dir and not self._docs:
            self._load_disk()
        missing: list[str] = []
        missing_keys: list[str] = []
        for text in texts:
            key = self._key(text)
            if key in self._docs:
                self.hits += 1
            else:
                missing.append(text)
                missing_keys.append(key)
        if missing:
            self.misses += len(missing)
            vectors = self.inner.embed_documents(missing)
            for key, vector in zip(missing_keys, vectors):
                self._docs[key] = vector
            if self._dir and len(missing) >= 8:
                self._save_disk()
        return [self._docs[self._key(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if not self.enabled:
            return self.inner.embed_query(text)
        key = self._key(text)
        cached = self._queries.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vector = self.inner.embed_query(text)
        self._queries[key] = vector
        return vector

    def info(self) -> dict[str, Any]:
        base = self.inner.info() if hasattr(self.inner, "info") else {}
        base.update({"cache_hits": self.hits, "cache_misses": self.misses, "cached": self.enabled})
        return base


def build_embedder(cfg: EmbeddingCfg, cache_dir: str | Path | None = None) -> CachedEmbedder:
    """按配置构造向量化器，``auto`` 优先 BGE、失败自动降级。"""
    provider = (cfg.provider or "auto").lower()
    inner: Embedder
    if provider == "bge":
        inner = BgeEmbedder(
            model_name=cfg.model,
            dim=cfg.dim,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
            query_prefix=cfg.query_prefix,
            doc_prefix=cfg.doc_prefix,
        )
    elif provider == "hashing":
        inner = HashingEmbedder(
            dim=cfg.dim,
            normalize=cfg.normalize,
            query_prefix=cfg.query_prefix,
        )
    else:  # auto
        try:
            candidate = BgeEmbedder(
                model_name=cfg.model,
                dim=cfg.dim,
                batch_size=cfg.batch_size,
                normalize=cfg.normalize,
                query_prefix=cfg.query_prefix,
                doc_prefix=cfg.doc_prefix,
            )
            candidate.embed_query("预热")  # 提前触发依赖/权重加载
            inner = candidate
        except Exception:
            inner = HashingEmbedder(
                dim=cfg.dim,
                normalize=cfg.normalize,
                query_prefix=cfg.query_prefix,
            )
    embedder = CachedEmbedder(inner, cache_dir=cache_dir, enabled=cfg.cache)
    return embedder


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = a[i]
        y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))
