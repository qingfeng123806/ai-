"""配置模型：TOML 为默认格式（Python 3.11+ 标准库解析），支持 JSON / YAML。

配置被切分为若干 section dataclass，A/B 实验通过 ``overrides`` 做过深度合并，
配置指纹用于把「一次实验」与「确切的一组参数」绑定，避免报告追溯时产生歧义。
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from .text import short_hash

CONFIG_SUFFIXES = (".toml", ".json", ".yaml", ".yml")

CHUNKERS = ("fixed", "recursive", "heading", "semantic")
EMBEDDERS = ("auto", "bge", "hashing")
STORES = ("auto", "memory", "chroma", "milvus")
RERANKERS = ("auto", "bge", "lexical", "none")
GENERATORS = ("auto", "openai", "offline")
REFUSAL_POLICIES = ("strict", "lenient")


class ConfigError(ValueError):
    """配置非法：字段未知、取值越界或引用缺失。"""


class _Section:
    """section 基类：字典互转 + 未知字段检测。"""

    @classmethod
    def _field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "_Section":
        data = dict(data or {})
        known = cls._field_names()
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"[{cls.__name__}] 存在未知字段 {unknown}；可用字段：{sorted(known)}"
            )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def key_diffs(self, other: "_Section") -> list[tuple[str, Any, Any]]:
        left, right = asdict(self), asdict(other)
        out: list[tuple[str, Any, Any]] = []
        for key in left:
            if left[key] != right.get(key):
                out.append((key, left[key], right.get(key)))
        return out


@dataclass
class ProjectCfg(_Section):
    name: str = "kb-rag-eval"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    db_path: str = "artifacts/eval.db"
    locale: str = "zh-CN"


@dataclass
class IngestCfg(_Section):
    paths: list[str] = field(default_factory=lambda: ["data/corpus"])
    chunker: str = "recursive"
    chunk_size: int = 420
    chunk_overlap: int = 60
    min_chunk_chars: int = 60
    max_chunk_chars: int = 1200
    heading_aware: bool = True
    semantic_threshold: float = 0.62
    include_globs: list[str] = field(
        default_factory=lambda: ["*.md", "*.markdown", "*.txt", "*.html", "*.htm", "*.pdf", "*.docx"]
    )


@dataclass
class EmbeddingCfg(_Section):
    provider: str = "auto"
    model: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache: bool = True
    query_prefix: str = ""
    doc_prefix: str = ""


@dataclass
class StoreCfg(_Section):
    provider: str = "auto"
    collection: str = "kb_eval"
    persist_dir: str = "artifacts/index"
    metric: str = "cosine"
    milvus_uri: str = "http://localhost:19530"


@dataclass
class RetrieveCfg(_Section):
    top_k: int = 6
    candidate_k: int = 24
    hybrid: bool = True
    rrf_k: int = 60
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    min_score: float = 0.0
    mmr_lambda: float = 0.0


@dataclass
class RerankCfg(_Section):
    provider: str = "auto"
    model: str = "BAAI/bge-reranker-base"
    top_n: int = 4
    batch_size: int = 16


@dataclass
class GenerateCfg(_Section):
    provider: str = "auto"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1"
    api_key_env: str = "RAG_EVAL_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    prompt: str = "grounded_cite_v1"
    require_citations: bool = True
    dedupe_sentences: bool = True
    refusal_policy: str = "strict"
    max_context_chars: int = 4000
    timeout_s: float = 60.0
    max_retries: int = 2
    cost_input_per_1k: float = 0.0
    cost_output_per_1k: float = 0.0


@dataclass
class EvaluateCfg(_Section):
    metrics: list[str] = field(default_factory=list)
    judge_provider: str = "auto"
    judge_model: str = ""
    judge_temperature: float = 0.0
    bootstrap_samples: int = 1000
    confidence: float = 0.95
    significance_alpha: float = 0.05
    ragas: bool = True
    deepeval: bool = True
    badcase_threshold: float = 0.6
    flag_rules: dict[str, float] = field(default_factory=dict)
    max_cases: int = 0
    retry_failed: bool = True


@dataclass
class ReportCfg(_Section):
    out_dir: str = "artifacts/reports"
    formats: list[str] = field(default_factory=lambda: ["html", "markdown", "csv", "json"])
    title: str = "RAG 评测报告"
    include_contexts: bool = True
    max_badcases: int = 30


@dataclass
class VariantCfg:
    """A/B 实验中的一个实验组。"""

    name: str
    description: str = ""
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VariantCfg":
        data = dict(data)
        unknown = sorted(set(data) - {"name", "description", "overrides"})
        if unknown:
            raise ConfigError(f"[variant] 存在未知字段 {unknown}")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ConfigError("每个 variant 必须有 name")
        overrides = data.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ConfigError(f"variant {name} 的 overrides 必须是 [section] 表")
        return cls(name=name, description=str(data.get("description") or ""), overrides=copy.deepcopy(overrides))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "overrides": copy.deepcopy(self.overrides)}


@dataclass
class ExperimentCfg:
    name: str
    dataset: str
    variants: list[VariantCfg]
    baseline: str = ""
    description: str = ""
    seed: int = 42
    output_dir: str = "artifacts/experiments"
    repeat: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentCfg":
        data = dict(data)
        known = {"name", "dataset", "variants", "baseline", "description", "seed", "output_dir", "repeat"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"[experiment] 存在未知字段 {unknown}")
        name = str(data.get("name") or "").strip()
        if not name:
            raise ConfigError("experiment.name 不能为空")
        variants = [VariantCfg.from_dict(v) for v in (data.get("variants") or [])]
        if len(variants) < 2:
            raise ConfigError("A/B 实验至少需要两个 variant")
        names = [v.name for v in variants]
        if len(set(names)) != len(names):
            raise ConfigError(f"variant 名称重复：{names}")
        dataset = str(data.get("dataset") or "").strip()
        if not dataset:
            raise ConfigError("experiment.dataset 不能为空")
        return cls(
            name=name,
            dataset=dataset,
            variants=variants,
            baseline=str(data.get("baseline") or ""),
            description=str(data.get("description") or ""),
            seed=int(data.get("seed", 42)),
            output_dir=str(data.get("output_dir") or "artifacts/experiments"),
            repeat=int(data.get("repeat", 1) or 1),
        )


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    ingest: IngestCfg = field(default_factory=IngestCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    store: StoreCfg = field(default_factory=StoreCfg)
    retrieve: RetrieveCfg = field(default_factory=RetrieveCfg)
    rerank: RerankCfg = field(default_factory=RerankCfg)
    generate: GenerateCfg = field(default_factory=GenerateCfg)
    evaluate: EvaluateCfg = field(default_factory=EvaluateCfg)
    report: ReportCfg = field(default_factory=ReportCfg)
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    SECTIONS = {
        "project": ProjectCfg,
        "ingest": IngestCfg,
        "embedding": EmbeddingCfg,
        "store": StoreCfg,
        "retrieve": RetrieveCfg,
        "rerank": RerankCfg,
        "generate": GenerateCfg,
        "evaluate": EvaluateCfg,
        "report": ReportCfg,
    }

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source_path: str = "") -> "Config":
        data = dict(data or {})
        cfg = cls(source_path=source_path)
        unknown = sorted(set(data) - set(cls.SECTIONS))
        if unknown:
            raise ConfigError(f"存在未知配置段 {unknown}；可用段：{sorted(cls.SECTIONS)}")
        for name, section_cls in cls.SECTIONS.items():
            if name in data:
                if not isinstance(data[name], dict):
                    raise ConfigError(f"[{name}] 必须是表/对象")
                setattr(cfg, name, section_cls.from_dict(data[name]))
        cfg.raw = copy.deepcopy(dict(data))
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"配置文件不存在：{path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".toml":
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
                raise ConfigError("解析 TOML 需要 Python 3.11+ 或安装 tomli") from None
            data = tomllib.loads(text)
        elif suffix == ".json":
            data = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError:
                raise ConfigError("解析 YAML 需要安装 PyYAML，或改用 .toml 配置") from None
            data = yaml.safe_load(text) or {}
        else:
            raise ConfigError(f"不支持的配置后缀 {suffix}，可用：{CONFIG_SUFFIXES}")
        return cls.from_dict(data, source_path=str(path))

    @classmethod
    def load_experiment(cls, path: str | Path) -> ExperimentCfg:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"实验配置不存在：{path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".toml":
            import tomllib

            data = tomllib.loads(text).get("experiment", {})
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
            data = data.get("experiment", data)
        else:
            raise ConfigError("实验配置仅支持 .toml / .json")
        exp = ExperimentCfg.from_dict(data)
        if not exp.baseline:
            exp.baseline = str(path.parent.parent / "baseline.toml") if path.parent.name == "experiments" else "configs/baseline.toml"
        return exp

    # -- 操作 -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in self.SECTIONS}

    def clone(self) -> "Config":
        return Config.from_dict(self.to_dict(), self.source_path)

    def apply_overrides(self, overrides: Mapping[str, Any]) -> "Config":
        """按 section 深度合并覆盖项，返回新配置。"""
        if not overrides:
            return self.clone()
        merged = self.to_dict()
        for section, values in overrides.items():
            if section not in self.SECTIONS:
                raise ConfigError(f"覆盖项含未知配置段 [{section}]")
            if not isinstance(values, dict):
                raise ConfigError(f"覆盖项 [{section}] 必须是表")
            base = merged.setdefault(section, {})
            base.update(copy.deepcopy(values))
        merged_cfg = Config.from_dict(merged, self.source_path)
        merged_cfg.raw = copy.deepcopy(merged)
        return merged_cfg

    def fingerprint(self) -> str:
        return short_hash(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True), 16)

    def diff(self, other: "Config") -> list[dict[str, Any]]:
        """返回两套配置之间的差异（供报告与会话记录展示）。"""
        left, right = self.to_dict(), other.to_dict()
        out: list[dict[str, Any]] = []
        for section in self.SECTIONS:
            for key, value in left[section].items():
                other_value = right.get(section, {}).get(key)
                if value != other_value:
                    out.append({"path": f"{section}.{key}", "left": value, "right": other_value})
        return out

    def resolve(self, path: str) -> Path:
        """相对路径统一相对工程根目录解析（以配置文件所在目录的上一级为根）。"""
        p = Path(path)
        if p.is_absolute():
            return p
        return (Path.cwd() / p).resolve()

    def api_key(self) -> str:
        return os.environ.get(self.generate.api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")

    # -- 校验 -------------------------------------------------------------
    def validate(self) -> None:
        errs: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errs.append(msg)

        check(self.ingest.chunker in CHUNKERS, f"ingest.chunker 必须是 {CHUNKERS}")
        check(self.embedding.provider in EMBEDDERS, f"embedding.provider 必须是 {EMBEDDERS}")
        check(self.embedding.dim > 0, "embedding.dim 必须为正整数")
        check(self.store.provider in STORES, f"store.provider 必须是 {STORES}")
        check(self.rerank.provider in RERANKERS, f"rerank.provider 必须是 {RERANKERS}")
        check(self.generate.provider in GENERATORS, f"generate.provider 必须是 {GENERATORS}")
        check(self.generate.refusal_policy in REFUSAL_POLICIES, f"generate.refusal_policy 必须是 {REFUSAL_POLICIES}")
        check(0.0 <= self.generate.temperature <= 2.0, "generate.temperature 必须在 [0,2]")
        check(self.retrieve.top_k >= 1, "retrieve.top_k 至少为 1")
        check(self.retrieve.candidate_k >= self.retrieve.top_k, "retrieve.candidate_k 不能小于 top_k")
        check(self.ingest.chunk_overlap < self.ingest.chunk_size, "ingest.chunk_overlap 必须小于 chunk_size")
        check(0.0 < self.ingest.chunk_size <= 8000, "ingest.chunk_size 需在 (0,8000]")
        check(self.rerank.top_n >= 1, "rerank.top_n 至少为 1")
        check(0 < self.evaluate.confidence < 1, "evaluate.confidence 需在 (0,1)")
        check(self.evaluate.bootstrap_samples >= 0, "evaluate.bootstrap_samples 不能为负")
        if errs:
            raise ConfigError("配置校验失败：\n- " + "\n- ".join(errs))
