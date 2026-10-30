"""领域模型：文档、切片、检索结果、测试用例、评测结果、运行记录。

所有模型都提供 ``to_dict`` / ``from_dict``，便于写入 JSONL、SQLite 与 HTTP API。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

SPLITS = ("core", "edge", "adversarial")
SPLIT_LABELS = {"core": "核心回归集", "edge": "边缘场景集", "adversarial": "对抗集"}
DIFFICULTIES = ("easy", "medium", "hard")


def _clean(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    return value


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "chars": len(self.text),
            "meta": _clean(self.meta),
        }


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    index: int = 0
    start: int = 0
    end: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "text": self.text,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "meta": _clean(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data.get("doc_id", "")),
            title=str(data.get("title", "")),
            source=str(data.get("source", "")),
            text=str(data.get("text", "")),
            index=int(data.get("index", 0) or 0),
            start=int(data.get("start", 0) or 0),
            end=int(data.get("end", 0) or 0),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class Retrieved:
    chunk: Chunk
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float | None = None
    rank: int = 0
    source_list: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    def to_dict(self, max_chars: int = 600) -> dict[str, Any]:
        text = self.chunk.text
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "title": self.chunk.title,
            "source": self.chunk.source,
            "text": text,
            "preview": text if len(text) <= max_chars else text[: max_chars - 1] + "…",
            "index": self.chunk.index,
            "rank": self.rank,
            "dense_score": round(self.dense_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "fusion_score": round(self.fusion_score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "score": round(self.score, 4),
            "sources": list(self.source_list),
        }


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_id: str = ""
    title: str = ""
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Generation:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [c.to_dict() for c in self.citations]
        return data


@dataclass
class CaseSpec:
    case_id: str
    question: str
    split: str = "core"
    reference_answer: str = ""
    reference_contexts: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = "human"
    difficulty: str = "medium"
    should_refuse: bool = False
    adversarial_type: str = ""
    language: str = "zh"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "split": self.split,
            "reference_answer": self.reference_answer,
            "reference_contexts": list(self.reference_contexts),
            "expected_keywords": list(self.expected_keywords),
            "tags": list(self.tags),
            "source": self.source,
            "difficulty": self.difficulty,
            "should_refuse": bool(self.should_refuse),
            "adversarial_type": self.adversarial_type,
            "language": self.language,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseSpec":
        raw_id = str(data.get("case_id") or data.get("id") or "").strip()
        question = str(data.get("question") or "").strip()
        if not question:
            raise ValueError(f"用例 {raw_id or '<no-id>'} 缺少 question")
        if not raw_id:
            from .text import short_hash

            raw_id = "auto-" + short_hash(question, 10)
        split = str(data.get("split") or "core").strip().lower()
        if split not in SPLITS:
            split = "core"
        return cls(
            case_id=raw_id,
            question=question,
            split=split,
            reference_answer=str(data.get("reference_answer") or data.get("reference") or "").strip(),
            reference_contexts=[str(x) for x in (data.get("reference_contexts") or []) if str(x).strip()],
            expected_keywords=[str(x) for x in (data.get("expected_keywords") or []) if str(x).strip()],
            tags=[str(x) for x in (data.get("tags") or [])],
            source=str(data.get("source") or "human"),
            difficulty=str(data.get("difficulty") or "medium"),
            should_refuse=bool(data.get("should_refuse", False)),
            adversarial_type=str(data.get("adversarial_type") or ""),
            language=str(data.get("language") or "zh"),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class LatencyBreakdown:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_ms": round(self.retrieval_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class CaseResult:
    case: CaseSpec
    answer: str = ""
    refused: bool = False
    citations: list[Citation] = field(default_factory=list)
    contexts: list[Retrieved] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""
    error: str = ""
    run_id: str = ""
    variant: str = ""

    @property
    def flagged(self) -> bool:
        return bool(self.details.get("flagged"))

    @property
    def flags(self) -> list[str]:
        return list(self.details.get("flags") or [])

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "variant": self.variant,
            "case": self.case.to_dict(),
            "case_id": self.case.case_id,
            "question": self.case.question,
            "split": self.case.split,
            "answer": self.answer,
            "refused": self.refused,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": [c.to_dict() for c in self.contexts],
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "details": self.details,
            "latency": self.latency.to_dict(),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "provider": self.provider,
            "error": self.error,
            "flagged": self.flagged,
            "flags": self.flags,
        }


@dataclass
class MetricScore:
    name: str
    value: float
    group: str = "generation"
    scope: str = "case"
    direction: str = "higher"
    n: int = 0
    ci_low: float = 0.0
    ci_high: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "group": self.group,
            "scope": self.scope,
            "direction": self.direction,
            "n": self.n,
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "description": self.description,
        }


@dataclass
class RunRecord:
    run_id: str
    name: str
    created_at: str
    dataset: str
    config_fingerprint: str
    config: dict[str, Any]
    n_cases: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)
    mode: str = "offline"
    status: str = "completed"
    experiment_id: str = ""
    variant: str = "baseline"
    duration_s: float = 0.0
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> float | None:
        for item in self.metrics:
            if item.get("name") == name:
                return float(item.get("value", 0.0))
        return None

    def headline(self) -> dict[str, float]:
        wanted = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "hallucination_rate",
            "citation_accuracy",
        ]
        out: dict[str, float] = {}
        for name in wanted:
            value = self.metric(name)
            if value is not None:
                out[name] = value
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_texts(items: Iterable[str]) -> str:
    return "\n".join(x for x in items if x)
