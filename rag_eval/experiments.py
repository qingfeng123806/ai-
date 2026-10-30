"""A/B 实验引擎：配置矩阵、独立索引、配对显著性检验、综合评分与冠军判定。

实验流程
--------
1. 以基线配置为底，按 variant 的 ``overrides`` 深度合并出各组配置；
2. 每组配置独立构建索引（切片参数不同 → 内容指纹不同 → 各自缓存）；
3. 同一批测试用例上跑完整评测，得到逐用例指标；
4. 以基线为对照，做逐指标配对 Bootstrap 检验，输出差值与 p 值；
5. 按关键质量指标的综合得分判定冠军，并给出「显著变好 / 显著变差」计数。

配对设计的价值：同一用例、同一提示词，只有配置不同，差值的方差远小于
独立样本比较，几十条用例就能看出趋势。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from . import db
from .config import Config, ExperimentCfg
from .dataset import dataset_stats, load_cases
from .metrics import specs_by_name
from .pipeline import RAGPipeline, build_index
from .runner import EvaluationRun, RunOptions, run_evaluation
from .stats import compare_series, mean
from .text import short_hash
from .types import SPLIT_LABELS

# 综合评分使用的关键指标（方向感知：越小越好的指标会被翻转）
COMPOSITE_METRICS = [
    "hit_rate",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_completeness",
    "citation_accuracy",
    "refusal_appropriateness",
    "hallucination_rate",
]


@dataclass
class VariantOutcome:
    name: str
    description: str
    overrides: dict[str, Any]
    error: str
    run: EvaluationRun | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "overrides": self.overrides,
            "error": self.error,
            "run_id": self.run.record.run_id if self.run else "",
            "mode": self.run.record.mode if self.run else "",
            "metrics": self.run.record.metrics if self.run else [],
            "n_cases": self.run.record.n_cases if self.run else 0,
        }


@dataclass
class ExperimentOutcome:
    experiment_id: str
    name: str
    created_at: str
    dataset: str
    baseline: str
    outcomes: list[VariantOutcome] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""

    def variant_map(self) -> dict[str, VariantOutcome]:
        return {o.name: o for o in self.outcomes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "baseline": self.baseline,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "comparisons": self.comparisons,
            "summary": self.summary,
            "report_path": self.report_path,
        }


def composite_score(record) -> float:
    """关键指标的方向归一化均值，用于粗略排序（显著性与效应量另看）。"""
    specs = specs_by_name()
    values: list[float] = []
    for name in COMPOSITE_METRICS:
        value = record.metric(name)
        if value is None:
            continue
        spec = specs.get(name)
        quality = value if (spec is None or spec.direction == "higher") else 1.0 - value
        values.append(max(0.0, min(1.0, quality)))
    return mean(values) if values else 0.0


def _series_by_case(run: EvaluationRun) -> dict[str, dict[str, float]]:
    return {r.case.case_id: dict(r.scores) for r in run.results}


def compare_results(left: EvaluationRun, right: EvaluationRun, alpha: float = 0.05, bootstrap_samples: int = 2000) -> dict[str, Any]:
    """两组逐用例结果做指标级配对对比。"""
    left_map = _series_by_case(left)
    right_map = _series_by_case(right)
    shared_cases = sorted(set(left_map) & set(right_map))
    specs = specs_by_name()
    metric_names = sorted(set().union(*(set(m) for m in left_map.values()), *(set(m) for m in right_map.values())))

    deltas: list[dict[str, Any]] = []
    for name in metric_names:
        spec = specs.get(name)
        if spec is None or spec.scope != "case":
            continue
        pairs = [
            (left_map[c][name], right_map[c][name])
            for c in shared_cases
            if name in left_map[c] and name in right_map[c]
        ]
        if len(pairs) < 2:
            continue
        l_vals = [p[0] for p in pairs]
        r_vals = [p[1] for p in pairs]
        result = compare_series(
            name,
            l_vals,
            r_vals,
            direction=spec.direction,
            alpha=alpha,
            samples=bootstrap_samples,
            seed=f"{left.record.run_id}:{right.record.run_id}:{name}",
        )
        deltas.append(result.to_dict())

    improved = [d for d in deltas if d["significant"] and d["better"] == "right"]
    degraded = [d for d in deltas if d["significant"] and d["better"] == "left"]
    case_diffs = _case_diffs(left_map, right_map, shared_cases)
    return {
        "left": {"run_id": left.record.run_id, "variant": left.record.variant, "name": left.record.name},
        "right": {"run_id": right.record.run_id, "variant": right.record.variant, "name": right.record.name},
        "n_cases": len(shared_cases),
        "deltas": deltas,
        "improved": improved,
        "degraded": degraded,
        "case_diffs": case_diffs,
        "composite": {"left": round(composite_score(left.record), 4), "right": round(composite_score(right.record), 4)},
    }


def _case_diffs(
    left_map: dict[str, dict[str, float]],
    right_map: dict[str, dict[str, float]],
    shared: Sequence[str],
) -> list[dict[str, Any]]:
    """按综合质量分找出变化最大的用例，供「答案对比」列表使用。"""
    out: list[dict[str, Any]] = []
    metrics = ["faithfulness", "answer_relevancy", "answer_completeness", "citation_accuracy", "context_recall"]

    def quality(scores: dict[str, float]) -> float:
        vals = [scores[m] for m in metrics if m in scores]
        return mean(vals) if vals else 0.0

    for case_id in shared:
        left_q = quality(left_map[case_id])
        right_q = quality(right_map[case_id])
        out.append(
            {
                "case_id": case_id,
                "left": round(left_q, 4),
                "right": round(right_q, 4),
                "delta": round(right_q - left_q, 4),
            }
        )
    out.sort(key=lambda item: -abs(item["delta"]))
    return out[:20]


def compare_records(cfg: Config, left, right) -> dict[str, Any]:
    """仅基于持久化记录的对比（无逐用例数据时退化为指标级差值）。"""
    specs = specs_by_name()
    deltas: list[dict[str, Any]] = []
    metrics = {m["name"]: m for m in left.metrics}
    for item in right.metrics:
        name = item["name"]
        if name not in metrics:
            continue
        spec = specs.get(name)
        l_val = float(metrics[name]["value"])
        r_val = float(item["value"])
        delta = r_val - l_val
        if abs(delta) < 1e-9:
            better = "tie"
        elif spec is not None and spec.direction == "lower":
            better = "right" if delta < 0 else "left"
        else:
            better = "right" if delta > 0 else "left"
        deltas.append(
            {
                "metric": name,
                "left": l_val,
                "right": r_val,
                "delta": round(delta, 4),
                "relative": round(delta / l_val, 4) if abs(l_val) > 1e-12 else 0.0,
                "better": better,
                "direction": spec.direction if spec else "higher",
                "significant": None,
                "p_value": None,
            }
        )
    return {
        "left": {"run_id": left.run_id, "variant": left.variant, "name": left.name},
        "right": {"run_id": right.run_id, "variant": right.variant, "name": right.name},
        "deltas": deltas,
        "n_cases": min(left.n_cases, right.n_cases),
        "composite": {"left": round(composite_score(left), 4), "right": round(composite_score(right), 4)},
        "mode": "record-only",
    }


def run_experiment(
    base_cfg: Config,
    exp: ExperimentCfg,
    cases: Sequence | None = None,
    progress: Any = None,
    persist: bool = True,
) -> ExperimentOutcome:
    """执行一次完整 A/B 实验。"""
    if cases is None:
        dataset_path = Path(exp.dataset)
        if not dataset_path.exists():
            fallback = Path("data/testsets/core.jsonl")
            if fallback.exists():
                dataset_path = fallback
        cases = load_cases(dataset_path)
    if not cases:
        raise ValueError("实验测试集为空")

    experiment_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{short_hash(exp.name + base_cfg.fingerprint(), 6)}"
    outcome = ExperimentOutcome(
        experiment_id=experiment_id,
        name=exp.name,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        dataset=exp.dataset,
        baseline=exp.variants[0].name,
    )

    if persist:
        db.init_db(base_cfg.project.db_path)

    pipelines: dict[str, RAGPipeline] = {}
    total_variants = len(exp.variants)
    for index, variant in enumerate(exp.variants, start=1):
        variant_cfg = base_cfg.apply_overrides(variant.overrides)
        fingerprint = variant_cfg.fingerprint()
        if progress:
            progress("variant_start", index, total_variants, variant.name)
        try:
            pipeline = pipelines.get(fingerprint)
            if pipeline is None:
                pipeline = RAGPipeline(variant_cfg, build_index(variant_cfg))
                pipelines[fingerprint] = pipeline
            options = RunOptions(
                name=f"{exp.name}::{variant.name}",
                dataset=exp.dataset,
                experiment_id=experiment_id,
                variant=variant.name,
                notes=variant.description,
                progress=(lambda i, total, result, _name=variant.name: progress("case", i, total, _name, result)) if progress else None,
                persist=persist,
            )
            run = run_evaluation(variant_cfg, list(cases), options, pipeline=pipeline)
        except Exception as exc:  # 单组失败不拖垮整个实验
            outcome.outcomes.append(
                VariantOutcome(
                    name=variant.name,
                    description=variant.description,
                    overrides=variant.overrides,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        # 记录与基线的配置差异，报告中可追溯「到底改了什么」
        run.record.metadata["overrides"] = variant.overrides
        outcome.outcomes.append(
            VariantOutcome(
                name=variant.name,
                description=variant.description,
                overrides=variant.overrides,
                error="",
                run=run,
            )
        )

    baseline_outcome = next((o for o in outcome.outcomes if o.run is not None), None)
    if baseline_outcome is not None:
        outcome.baseline = baseline_outcome.name
        for item in outcome.outcomes:
            if item.run is None or item.name == baseline_outcome.name:
                continue
            comparison = compare_results(
                baseline_outcome.run,
                item.run,
                alpha=base_cfg.evaluate.significance_alpha,
                bootstrap_samples=max(500, base_cfg.evaluate.bootstrap_samples),
            )
            comparison["variant"] = item.name
            outcome.comparisons.append(comparison)

    outcome.summary = _summarize(outcome)

    if persist:
        report_dir = Path(exp.output_dir) / experiment_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "experiment.json"
        report_path.write_text(json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        outcome.report_path = str(report_path)
        db.save_experiment(
            base_cfg.project.db_path,
            experiment_id=experiment_id,
            name=exp.name,
            created_at=outcome.created_at,
            dataset=exp.dataset,
            config=base_cfg.to_dict(),
            variants=[o.to_dict() for o in outcome.outcomes],
            status="completed" if all(o.run for o in outcome.outcomes) else "partial",
            summary=outcome.summary,
        )
    return outcome


def _summarize(outcome: ExperimentOutcome) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in outcome.outcomes:
        if item.run is None:
            rows.append({"variant": item.name, "error": item.error, "composite": 0.0})
            continue
        record = item.run.record
        rows.append(
            {
                "variant": item.name,
                "description": item.description,
                "composite": round(composite_score(record), 4),
                "mode": record.mode,
                "duration_s": record.duration_s,
                "headline": {k: round(v, 4) for k, v in record.headline().items()},
                "n_cases": record.n_cases,
                "overrides": item.overrides,
                "run_id": record.run_id,
            }
        )
    valid = [row for row in rows if "error" not in row]
    winner = max(valid, key=lambda row: row["composite"])["variant"] if valid else ""
    improved = 0
    degraded = 0
    for comparison in outcome.comparisons:
        improved += len(comparison.get("improved", []))
        degraded += len(comparison.get("degraded", []))
    return {
        "winner": winner,
        "ranking": sorted(valid, key=lambda row: -row["composite"]),
        "rows": rows,
        "significant_improvements": improved,
        "significant_regressions": degraded,
        "baseline": outcome.baseline,
        "n_variants": len(outcome.outcomes),
        "split_labels": SPLIT_LABELS,
    }


def load_experiment(outcome_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(outcome_path).read_text(encoding="utf-8"))
