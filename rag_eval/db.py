"""SQLite 持久层：运行记录、逐用例结果、实验、产物登记。

单文件数据库让「评测 → 看板 → 导出」在无外部服务时也能闭环；
WAL 模式支持评测写入与看板读取并发。所有 JSON 字段统一序列化，
避免 schema 随指标增减而迁移。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .types import CaseResult, RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    config_fingerprint TEXT,
    config_json TEXT,
    n_cases INTEGER DEFAULT 0,
    split_counts_json TEXT,
    metrics_json TEXT,
    providers_json TEXT,
    mode TEXT,
    status TEXT,
    experiment_id TEXT,
    variant TEXT,
    duration_s REAL,
    notes TEXT,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    split TEXT,
    flagged INTEGER DEFAULT 0,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE INDEX IF NOT EXISTS idx_case_run ON case_results(run_id);
CREATE INDEX IF NOT EXISTS idx_case_flagged ON case_results(run_id, flagged);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    dataset TEXT,
    config_json TEXT,
    variants_json TEXT,
    status TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    path TEXT NOT NULL,
    kind TEXT,
    created_at TEXT
);
"""


@contextmanager
def _connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)  # 幂等：保证任何读接口在空库上也不会因缺表报错
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> Path:
    path = Path(db_path)
    with _connect(path) as conn:
        conn.executescript(SCHEMA)
    return path


def save_run(db_path: str | Path, record: RunRecord, results: Sequence[CaseResult] = ()) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO runs (
                run_id, name, created_at, dataset, config_fingerprint, config_json, n_cases,
                split_counts_json, metrics_json, providers_json, mode, status, experiment_id,
                variant, duration_s, notes, metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                name=excluded.name, metrics_json=excluded.metrics_json, n_cases=excluded.n_cases,
                split_counts_json=excluded.split_counts_json, providers_json=excluded.providers_json,
                mode=excluded.mode, status=excluded.status, duration_s=excluded.duration_s,
                notes=excluded.notes, metadata_json=excluded.metadata_json
            """,
            (
                record.run_id,
                record.name,
                record.created_at,
                record.dataset,
                record.config_fingerprint,
                json.dumps(record.config, ensure_ascii=False),
                record.n_cases,
                json.dumps(record.split_counts, ensure_ascii=False),
                json.dumps(record.metrics, ensure_ascii=False),
                json.dumps(record.providers, ensure_ascii=False),
                record.mode,
                record.status,
                record.experiment_id,
                record.variant,
                record.duration_s,
                record.notes,
                json.dumps(record.metadata, ensure_ascii=False),
            ),
        )
        if results:
            rows = [
                (
                    record.run_id,
                    item.case.case_id,
                    item.case.split,
                    1 if item.flagged else 0,
                    json.dumps(item.to_dict(), ensure_ascii=False),
                )
                for item in results
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO case_results (run_id, case_id, split, flagged, payload_json) VALUES (?,?,?,?,?)",
                rows,
            )


def update_run_status(db_path: str | Path, run_id: str, status: str, notes: str = "") -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE runs SET status=?, notes=? WHERE run_id=?", (status, notes, run_id))


def list_runs(db_path: str | Path, limit: int = 100, experiment_id: str | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        if experiment_id:
            cursor = conn.execute(
                "SELECT * FROM runs WHERE experiment_id=? ORDER BY created_at DESC LIMIT ?",
                (experiment_id, limit),
            )
        else:
            cursor = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [_row_to_run(row) for row in cursor.fetchall()]


def get_run(db_path: str | Path, run_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return _row_to_run(row) if row else None


def find_run_by_name(db_path: str | Path, name: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE name=? ORDER BY created_at DESC LIMIT 1", (name,)
        ).fetchone()
        return _row_to_run(row) if row else None


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    def load(value: Any, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    return {
        "run_id": row["run_id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "dataset": row["dataset"],
        "config_fingerprint": row["config_fingerprint"],
        "config": load(row["config_json"], {}),
        "n_cases": row["n_cases"],
        "split_counts": load(row["split_counts_json"], {}),
        "metrics": load(row["metrics_json"], []),
        "providers": load(row["providers_json"], {}),
        "mode": row["mode"],
        "status": row["status"],
        "experiment_id": row["experiment_id"],
        "variant": row["variant"],
        "duration_s": row["duration_s"],
        "notes": row["notes"],
        "metadata": load(row["metadata_json"], {}),
    }


def get_case_results(
    db_path: str | Path,
    run_id: str,
    split: str | None = None,
    flagged_only: bool = False,
    limit: int = 0,
) -> list[dict[str, Any]]:
    query = "SELECT payload_json FROM case_results WHERE run_id=?"
    params: list[Any] = [run_id]
    if split:
        query += " AND split=?"
        params.append(split)
    if flagged_only:
        query += " AND flagged=1"
    query += " ORDER BY case_id"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            out.append(json.loads(row["payload_json"]))
        except json.JSONDecodeError:
            continue
    return out


def get_case_result(db_path: str | Path, run_id: str, case_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM case_results WHERE run_id=? AND case_id=?", (run_id, case_id)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["payload_json"])
    except json.JSONDecodeError:
        return None


def delete_run(db_path: str | Path, run_id: str) -> int:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM case_results WHERE run_id=?", (run_id,))
        cursor = conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        return cursor.rowcount


def save_experiment(
    db_path: str | Path,
    experiment_id: str,
    name: str,
    created_at: str,
    dataset: str,
    config: dict[str, Any],
    variants: list[dict[str, Any]],
    status: str = "running",
    summary: dict[str, Any] | None = None,
) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            """
            INSERT INTO experiments (experiment_id, name, created_at, dataset, config_json, variants_json, status, summary_json)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                status=excluded.status, summary_json=excluded.summary_json, variants_json=excluded.variants_json
            """,
            (
                experiment_id,
                name,
                created_at,
                dataset,
                json.dumps(config, ensure_ascii=False),
                json.dumps(variants, ensure_ascii=False),
                status,
                json.dumps(summary or {}, ensure_ascii=False),
            ),
        )


def list_experiments(db_path: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        rows = conn.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for row in rows:
        out.append(
            {
                "experiment_id": row["experiment_id"],
                "name": row["name"],
                "created_at": row["created_at"],
                "dataset": row["dataset"],
                "config": json.loads(row["config_json"] or "{}"),
                "variants": json.loads(row["variants_json"] or "[]"),
                "status": row["status"],
                "summary": json.loads(row["summary_json"] or "{}"),
            }
        )
    return out


def get_experiment(db_path: str | Path, experiment_id: str) -> dict[str, Any] | None:
    for item in list_experiments(db_path, limit=500):
        if item["experiment_id"] == experiment_id:
            item["runs"] = list_runs(db_path, experiment_id=experiment_id)
            return item
    return None


def register_artifact(db_path: str | Path, run_id: str, path: str, kind: str, created_at: str) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO artifacts (run_id, path, kind, created_at) VALUES (?,?,?,?)",
            (run_id, path, kind, created_at),
        )


def list_artifacts(db_path: str | Path, run_id: str) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        rows = conn.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
    return [dict(row) for row in rows]


def overview(db_path: str | Path) -> dict[str, Any]:
    """看板首页用的汇总：运行数、最新运行、指标趋势。"""
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        runs = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 200").fetchall()
        experiments = conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()["n"]
        cases = conn.execute("SELECT COUNT(*) AS n FROM case_results").fetchone()["n"]
    parsed = [_row_to_run(row) for row in runs]
    trend_metrics = ["faithfulness", "answer_relevancy", "context_precision", "hallucination_rate", "latency_p95"]
    trend = []
    for run in sorted(parsed, key=lambda r: r["created_at"]):
        point = {"run_id": run["run_id"], "name": run["name"], "created_at": run["created_at"], "variant": run["variant"]}
        for metric in trend_metrics:
            for item in run["metrics"]:
                if item.get("name") == metric:
                    point[metric] = item.get("value")
                    break
        trend.append(point)
    return {
        "runs": len(parsed),
        "experiments": experiments,
        "case_results": cases,
        "latest_run": parsed[0] if parsed else None,
        "trend": trend,
    }
