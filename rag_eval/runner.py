"""评测运行器：把流水线、判官、指标与持久层串成一次可追溯的 run。

一次 run 的产物
--------------
1. ``RunRecord``：配置指纹、后端清单、逐项聚合指标（含置信区间）、耗时；
2. 逐用例结果（JSONL + SQLite）：答案、引用、检索片段、逐句支撑度、坏例标记；
3. 可选的外部框架复算结果（RAGAs / DeepEval），单独成组并标注来源。

任何单用例异常都会被记录为 ``error`` 而不是中断整批评测——评测系统本身要比
被评测系统更稳。
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from . import db
from .adapters import DeepEvalAdapter, RagasAdapter, adapter_statuses
from .config import Config
from .dataset import DatasetStats, dataset_stats
from .generate import Answerer
from .judge import Judge
from .llm import build_llm
from .metrics import (
    EvalContext,
    aggregate_metrics,
    default_metric_names,
    evaluate_case,
    metric_series,
    specs_by_name,
    validate_metric_names,
)
from .pipeline import RAGPipeline, build_index
from .stats import bootstrap_ci, mean
from .text import build_idf, short_hash
from .types import CaseResult, CaseSpec, LatencyBreakdown, MetricScore, RunRecord

ProgressFn = Callable[[int, int, CaseResult], None]


@dataclass
class RunOptions:
    name: str = ""
    dataset: str = ""
    splits: list[str] | None = None
    max_cases: int = 0
    seed: int = 42
    experiment_id: str = ""
    variant: str = "baseline"
    notes: str = ""
    persist: bool = True
    use_judge: bool = True
    use_adapters: bool = True
    progress: ProgressFn | None = None


@dataclass
class EvaluationRun:
    record: RunRecord
    results: list[CaseResult]
    cases: list[CaseSpec]
    dataset_stats: DatasetStats | None = None

    @property
    def flagged(self) -> list[CaseResult]:
        return [r for r in self.results if r.flagged]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_run_id(cfg: Config, name: str, variant: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{variant}-{short_hash(cfg.fingerprint() + name + stamp, 6)}"


def resolve_metric_names(cfg: Config, use_judge: bool = True) -> list[str]:
    requested = list(cfg.evaluate.metrics) or default_metric_names()
    valid, unknown = validate_metric_names(requested)
    if unknown:
        raise ValueError(f"配置中的指标不存在：{unknown}；可用指标：{sorted(specs_by_name())}")
    if not use_judge:
        valid = [n for n in valid if not n.startswith("judge_")]
    if not valid:
        raise ValueError("指标列表为空，无法评测")
    return valid


def _judge_llm(cfg: Config):
    provider = (cfg.evaluate.judge_provider or "auto").lower()
    if provider == "offline":
        return None
    gen_cfg = cfg.generate
    if cfg.evaluate.judge_model:
        gen_cfg = replace(gen_cfg, model=cfg.evaluate.judge_model)
    gen_cfg = replace(
        gen_cfg,
        provider="openai",
        temperature=cfg.evaluate.judge_temperature,
        max_tokens=400,
    )
    return build_llm(gen_cfg)


def run_evaluation(
    cfg: Config,
    cases: Sequence[CaseSpec],
    options: RunOptions | None = None,
    pipeline: RAGPipeline | None = None,
) -> EvaluationRun:
    options = options or RunOptions()
    if not cases:
        raise ValueError("测试集为空")
    started = time.perf_counter()
    cfg = cfg if not options.seed else cfg

    pipeline = pipeline or RAGPipeline(cfg, build_index(cfg))
    metric_names = resolve_metric_names(cfg, use_judge=options.use_judge)
    idf = build_idf([c.text for c in pipeline.index.chunks])

    judge = Judge(cfg.evaluate, _judge_llm(cfg)) if options.use_judge else None
    run_id = make_run_id(cfg, options.name or options.dataset, options.variant)
    results: list[CaseResult] = []
    total = len(cases)

    for index, case in enumerate(cases, start=1):
        result = _evaluate_single(cfg, pipeline, case, idf, judge, run_id, options.variant, metric_names)
        results.append(result)
        if options.progress is not None:
            options.progress(index, total, result)

    # 外部框架复算（可选，失败不影响主结果）
    external = _run_adapters(cfg, list(cases), results, options)
    if external:
        _attach_external(results, external)

    metrics = aggregate_metrics(
        results,
        metric_names,
        bootstrap_samples=cfg.evaluate.bootstrap_samples,
        confidence=cfg.evaluate.confidence,
        seed=cfg.project.seed,
    )
    metrics.extend(_external_metric_scores(results, external, cfg))
    duration = time.perf_counter() - started
    stats = dataset_stats(list(cases))
    split_counts = {k: v for k, v in stats.splits.items() if v}

    record = RunRecord(
        run_id=run_id,
        name=options.name or f"{Path(options.dataset).stem or 'suite'}-{options.variant}",
        created_at=now_iso(),
        dataset=options.dataset,
        config_fingerprint=cfg.fingerprint(),
        config=cfg.to_dict(),
        n_cases=len(cases),
        split_counts=split_counts,
        metrics=[m.to_dict() for m in metrics],
        providers=pipeline.providers(),
        mode=pipeline.mode,
        status="completed",
        experiment_id=options.experiment_id,
        variant=options.variant,
        duration_s=round(duration, 2),
        notes=options.notes,
        metadata={
            "index": pipeline.index.info(),
            "judge": judge.info() if judge is not None else {"source": "disabled", "available": False, "model": ""},
            "adapters": adapter_statuses(cfg) if options.use_adapters else [],
            "dataset": stats.to_dict(),
            "metric_groups": sorted({m.group for m in metrics}),
            "config_diff_from_baseline": [],
        },
    )

    if options.persist:
        db.init_db(cfg.project.db_path)
        db.save_run(cfg.project.db_path, record, results)
        artifact = write_run_artifact(cfg, record, results)
        db.register_artifact(cfg.project.db_path, record.run_id, str(artifact), "results.jsonl", record.created_at)
    return EvaluationRun(record=record, results=results, cases=list(cases), dataset_stats=stats)


def _evaluate_single(
    cfg: Config,
    pipeline: RAGPipeline,
    case: CaseSpec,
    idf: dict[str, float],
    judge: Judge | None,
    run_id: str,
    variant: str,
    metric_names: Sequence[str],
) -> CaseResult:
    started = time.perf_counter()
    result = CaseResult(case=case, run_id=run_id, variant=variant)
    contexts: list[Retrieved] = []
    try:
        query = pipeline.query(case.question)
        result.answer = query.generation.answer
        result.refused = query.generation.refused
        result.citations = query.generation.citations
        result.latency = query.latency
        result.tokens_in = query.generation.prompt_tokens
        result.tokens_out = query.generation.completion_tokens
        result.cost_usd = query.generation.cost_usd
        result.model = query.generation.model
        result.provider = query.generation.provider
        contexts = list(query.contexts)
    except Exception as exc:  # 单例失败不影响整批
        result.error = f"{type(exc).__name__}: {exc}"
        result.details["traceback"] = traceback.format_exc(limit=3)
        result.latency = LatencyBreakdown(total_ms=(time.perf_counter() - started) * 1000)
    result.contexts = contexts

    ctx = EvalContext(case=case, idf=idf, cfg=cfg)
    if judge is not None:
        judge_output = judge.evaluate(case, result.answer, contexts, idf, refused=result.refused)
        ctx.judge_scores = judge_output.scores
        result.details["judge"] = judge_output.to_dict()

    scores, details = evaluate_case(result, ctx, metric_names, cfg.evaluate.flag_rules)
    result.scores = scores
    merged = dict(result.details)
    merged.update(details)
    result.details = merged
    if not result.answer.strip() and not result.error:
        result.error = "empty answer"
    return result


def _run_adapters(
    cfg: Config,
    cases: list[CaseSpec],
    results: list[CaseResult],
    options: RunOptions,
) -> dict[str, dict[str, dict[str, float]]]:
    if not options.use_adapters:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for adapter in (RagasAdapter(cfg), DeepEvalAdapter(cfg)):
        if not adapter.available:
            continue
        try:
            scores = adapter.evaluate(cases, results)
        except Exception:
            continue
        if scores:
            out[adapter.name] = scores
    return out


def _attach_external(results: list[CaseResult], external: dict[str, dict[str, dict[str, float]]]) -> None:
    for result in results:
        case_id = result.case.case_id
        payload: dict[str, Any] = {}
        for framework, mapping in external.items():
            if case_id in mapping:
                payload[framework] = mapping[case_id]
        if payload:
            result.details["external"] = payload


def _external_metric_scores(
    results: list[CaseResult],
    external: dict[str, dict[str, dict[str, float]]],
    cfg: Config,
) -> list[MetricScore]:
    out: list[MetricScore] = []
    for framework in external:
        names = sorted({name for mapping in external[framework].values() for name in mapping})
        for name in names:
            values = [
                float(mapping.get(name))
                for mapping in external[framework].values()
                if mapping.get(name) is not None
            ]
            if not values:
                continue
            low, high = bootstrap_ci(
                values,
                samples=cfg.evaluate.bootstrap_samples,
                confidence=cfg.evaluate.confidence,
                seed=f"{framework}:{name}",
            )
            out.append(
                MetricScore(
                    name=f"{framework}_{name}",
                    value=mean(values),
                    group="external",
                    scope="case",
                    direction="higher",
                    n=len(values),
                    ci_low=low,
                    ci_high=high,
                    description=f"{framework} 复算的 {name}（外部框架口径，仅作交叉验证）",
                )
            )
    return out


def write_run_artifact(cfg: Config, record: RunRecord, results: Sequence[CaseResult]) -> Path:
    out_dir = Path(cfg.project.artifacts_dir) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.run_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    (out_dir / f"{record.run_id}.meta.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def compare_runs(cfg: Config, left: RunRecord, right: RunRecord) -> dict[str, Any]:
    """基于已持久化的两次 run 做指标级对比（供 CLI / 看板复用）。"""
    from .experiments import compare_records

    return compare_records(cfg, left, right)
