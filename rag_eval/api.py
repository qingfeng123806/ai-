"""HTTP 接口层（传输无关）：同一份处理函数同时服务于标准库服务器与 FastAPI。

为什么不用框架耦合实现：评测系统的部署环境差异很大（内网离线、Docker、
Serverless），把业务处理写成纯函数 ``dispatch(method, path, query, body)``，
就可以在无第三方依赖时用 ``http.server`` 跑起来，装了 FastAPI 时再挂载同一套路由，
两条路径的行为完全一致，也不会因为换框架而重写业务逻辑。
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__, db
from .config import Config
from .dataset import dataset_stats, load_cases
from .experiments import compare_records, run_experiment, load_experiment
from .metrics import all_specs, default_metric_names
from .pipeline import describe_environment
from .report import collect as collect_report
from .report import write_report
from .runner import RunOptions, run_evaluation
from .types import SPLIT_LABELS

STATIC_DIR = Path(__file__).parent / "static"
SAFE_STATIC = {".html", ".js", ".css", ".svg", ".png", ".ico", ".woff2", ".map"}


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> "Response":
        return cls(
            status=status,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    @classmethod
    def error(cls, message: str, status: int = 400) -> "Response":
        return cls.json({"error": message, "status": status}, status=status)

    @classmethod
    def file(cls, path: Path, download_name: str = "", content_type: str = "") -> "Response":
        data = path.read_bytes()
        ctype = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        headers = {}
        if download_name:
            headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        return cls(status=200, body=data, content_type=ctype, headers=headers)

    @classmethod
    def text(cls, text: str, content_type: str = "text/plain; charset=utf-8") -> "Response":
        return cls(status=200, body=text.encode("utf-8"), content_type=content_type)


@dataclass
class Job:
    job_id: str
    kind: str
    status: str = "running"       # running | done | failed
    progress: float = 0.0
    message: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    finished_at: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """后台评测任务：看板上点「运行评测」不会阻塞页面。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def create(self, kind: str) -> Job:
        job = Job(job_id=f"job-{uuid.uuid4().hex[:8]}", kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


class Api:
    def __init__(self, cfg: Config, root: Path | None = None, default_dataset: str = "data/testsets/core.jsonl"):
        self.cfg = cfg
        self.root = Path(root or Path.cwd())
        self.default_dataset = default_dataset
        self.jobs = JobManager()
        db.init_db(cfg.project.db_path)

    # -- 路由 ------------------------------------------------------------
    def dispatch(self, method: str, raw_path: str, query: dict[str, list[str]] | None = None, body: bytes | None = None) -> Response:
        try:
            # 同时支持「调用方解析好 query 传入」与「把 ?a=b 含在 path 里」两种用法
            merged_query = dict(parse_qs(urlparse(raw_path).query))
            for key, values in (query or {}).items():
                merged_query[key] = list(values)
            return self._dispatch(method.upper(), raw_path, merged_query, body or b"")
        except Exception as exc:  # 任何未捕获异常都返回结构化错误，避免服务器 500 空白页
            return Response.json(
                {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=2)},
                status=500,
            )

    def _dispatch(self, method: str, raw_path: str, query: dict[str, list[str]], body: bytes) -> Response:
        parsed = urlparse(raw_path)
        path = unquote(parsed.path)
        parts = [p for p in path.split("/") if p]

        if method == "GET" and (path == "/" or path == ""):
            return self._static("index.html")
        if method == "GET" and parts and parts[0] == "static":
            return self._static("/".join(parts[1:]))
        if not parts or parts[0] != "api":
            if method == "GET":
                return self._static("index.html")  # 前端使用 hash 路由，其余路径回落到首页
            return Response.error("未知路径", 404)
        return self._api(method, parts[1:], query, body)

    def _api(self, method: str, parts: list[str], query: dict[str, list[str]], body: bytes) -> Response:
        if not parts:
            return Response.json({"name": "kb-rag-eval", "version": __version__, "endpoints": self._endpoints()})

        head = parts[0]
        if head == "health" and method == "GET":
            return self._health()
        if head == "overview" and method == "GET":
            return Response.json(db.overview(self.cfg.project.db_path))
        if head == "runs":
            return self._runs(method, parts[1:], query)
        if head == "compare" and method == "GET":
            return self._compare(query)
        if head == "experiments":
            return self._experiments(method, parts[1:], query, body)
        if head == "metrics" and method == "GET":
            return self._metrics()
        if head == "datasets" and method == "GET":
            return self._datasets()
        if head == "report" and method == "GET":
            return self._report(query)
        if head == "jobs" and method == "GET":
            if len(parts) > 1:
                job = self.jobs.get(parts[1])
                return Response.json(job.to_dict()) if job else Response.error("任务不存在", 404)
            return Response.json({"jobs": [j.to_dict() for j in self.jobs.list()[:20]]})
        if head == "eval" and method == "POST":
            return self._submit_eval(body)
        if head == "config" and method == "GET":
            return Response.json({"config": self.cfg.to_dict(), "fingerprint": self.cfg.fingerprint()})
        return Response.error(f"未知接口：/{'/'.join(parts)}", 404)

    def _endpoints(self) -> list[str]:
        return [
            "GET /api/health",
            "GET /api/overview",
            "GET /api/runs",
            "GET /api/runs/{run_id}",
            "GET /api/runs/{run_id}/cases",
            "GET /api/runs/{run_id}/cases/{case_id}",
            "GET /api/compare?left=&right=",
            "GET /api/experiments",
            "GET /api/experiments/{experiment_id}",
            "GET /api/metrics",
            "GET /api/datasets",
            "GET /api/report?run_id=&format=",
            "POST /api/eval",
            "GET /api/jobs/{job_id}",
            "GET /api/config",
        ]

    # -- 各个接口 --------------------------------------------------------
    def _health(self) -> Response:
        env = describe_environment(self.cfg)
        warnings: list[str] = []
        if env["embedding"]["fallback"]:
            warnings.append("BGE 模型不可用，已降级为哈希向量化（离线可用）")
        if env["llm"]["fallback"]:
            warnings.append("未配置 LLM API Key，已降级为抽取式生成（离线可用）")
        if not env["store"]["chroma_available"]:
            warnings.append("Chroma 未安装，使用内存向量库（结果持久化为本地文件）")
        return Response.json(
            {
                "status": "ok",
                "version": __version__,
                "mode": "offline" if len(warnings) >= 3 else "hybrid",
                "environment": env,
                "warnings": warnings,
                "db_path": self.cfg.project.db_path,
                "config_fingerprint": self.cfg.fingerprint(),
                "runs": len(db.list_runs(self.cfg.project.db_path, limit=1000)),
            }
        )

    def _runs(self, method: str, parts: list[str], query: dict[str, list[str]]) -> Response:
        if method == "GET" and not parts:
            limit = int(query.get("limit", ["50"])[0] or 50)
            experiment_id = query.get("experiment_id", [""])[0] or None
            runs = db.list_runs(self.cfg.project.db_path, limit=limit, experiment_id=experiment_id)
            return Response.json(
                {
                    "runs": [
                        {
                            "run_id": r["run_id"],
                            "name": r["name"],
                            "created_at": r["created_at"],
                            "dataset": r["dataset"],
                            "n_cases": r["n_cases"],
                            "mode": r["mode"],
                            "status": r["status"],
                            "variant": r["variant"],
                            "experiment_id": r["experiment_id"],
                            "duration_s": r["duration_s"],
                            "split_counts": r["split_counts"],
                            "providers": r["providers"],
                            "headline": _headline(r["metrics"]),
                            "badcases": _count_badcases(self.cfg.project.db_path, r["run_id"]),
                        }
                        for r in runs
                    ]
                }
            )
        if method == "GET" and len(parts) >= 1:
            run_id = parts[0]
            record = db.get_run(self.cfg.project.db_path, run_id)
            if record is None:
                return Response.error(f"运行记录不存在：{run_id}", 404)
            if len(parts) == 1:
                payload = collect_report(self.cfg, run_id, max_badcases=self.cfg.report.max_badcases)
                return Response.json(
                    {
                        "run": record,
                        "metrics": payload["metrics"],
                        "scorecard": payload["scorecard"],
                        "splits": payload["splits"],
                        "badcase_total": payload["badcase_total"],
                        "badcases": payload["badcases"],
                    }
                )
            if len(parts) >= 2 and parts[1] == "cases":
                if len(parts) >= 3:
                    case = db.get_case_result(self.cfg.project.db_path, run_id, parts[2])
                    return Response.json(case) if case else Response.error("用例不存在", 404)
                split = query.get("split", [""])[0] or None
                flagged = query.get("flagged", [""])[0] in ("1", "true", "yes")
                limit = int(query.get("limit", ["200"])[0] or 200)
                cases = db.get_case_results(self.cfg.project.db_path, run_id, split=split, flagged_only=flagged, limit=limit)
                return Response.json({"cases": cases, "count": len(cases)})
        if method == "DELETE" and len(parts) == 1:
            deleted = db.delete_run(self.cfg.project.db_path, parts[0])
            return Response.json({"deleted": deleted})
        return Response.error("不支持的运行接口", 404)

    def _compare(self, query: dict[str, list[str]]) -> Response:
        left_id = query.get("left", [""])[0]
        right_id = query.get("right", [""])[0]
        if not left_id or not right_id:
            return Response.error("需要 left 与 right 两个 run_id")
        left = db.get_run(self.cfg.project.db_path, left_id)
        right = db.get_run(self.cfg.project.db_path, right_id)
        if left is None or right is None:
            return Response.error("运行记录不存在", 404)

        # 有逐用例数据时做配对显著性检验，否则退化为指标级差值
        left_cases = db.get_case_results(self.cfg.project.db_path, left_id)
        right_cases = db.get_case_results(self.cfg.project.db_path, right_id)
        if left_cases and right_cases:
            comparison = compare_persisted(self.cfg, left, right, left_cases, right_cases)
        else:
            comparison = compare_records(self.cfg, _dict_record(left), _dict_record(right))
        comparison["left_run"] = {k: left[k] for k in ("run_id", "name", "variant", "n_cases", "mode", "created_at")}
        comparison["right_run"] = {k: right[k] for k in ("run_id", "name", "variant", "n_cases", "mode", "created_at")}
        return Response.json(comparison)

    def _experiments(self, method: str, parts: list[str], query: dict[str, list[str]], body: bytes) -> Response:
        if method == "GET" and not parts:
            items = db.list_experiments(self.cfg.project.db_path)
            return Response.json({"experiments": items})
        if method == "GET" and len(parts) == 1:
            experiment = db.get_experiment(self.cfg.project.db_path, parts[0])
            return Response.json(experiment) if experiment else Response.error("实验不存在", 404)
        if method == "POST" and not parts:
            payload = _parse_json(body)
            path = str(payload.get("experiment") or "").strip()
            if not path:
                return Response.error("缺少 experiment 配置路径")
            exp = Config.load_experiment(self.root / path)
            job = self.jobs.create("experiment")
            dataset = str(payload.get("dataset") or exp.dataset)

            def worker() -> None:
                try:
                    cases = load_cases(self.root / dataset)
                    outcome = run_experiment(self.cfg, exp, cases, persist=True)
                    self.jobs.update(
                        job.job_id,
                        status="done",
                        progress=1.0,
                        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                        result={
                            "experiment_id": outcome.experiment_id,
                            "winner": outcome.summary.get("winner"),
                            "report_path": outcome.report_path,
                        },
                        message="实验完成",
                    )
                except Exception as exc:
                    self.jobs.update(
                        job.job_id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    )

            threading.Thread(target=worker, daemon=True).start()
            return Response.json({"job_id": job.job_id, "status": "running"}, status=202)
        return Response.error("不支持实验接口", 404)

    def _metrics(self) -> Response:
        specs = [
            {
                "name": s.name,
                "group": s.group,
                "scope": s.scope,
                "direction": s.direction,
                "description": s.description,
                "unit": s.unit,
            }
            for s in all_specs()
        ]
        return Response.json({"metrics": specs, "defaults": default_metric_names()})

    def _datasets(self) -> Response:
        data_dir = self.root / "data" / "testsets"
        items: list[dict[str, Any]] = []
        if data_dir.exists():
            for path in sorted(data_dir.glob("*.jsonl")):
                try:
                    cases = load_cases(path)
                    stats = dataset_stats(cases)
                    items.append(
                        {
                            "path": path.relative_to(self.root).as_posix(),
                            "name": path.stem,
                            "count": stats.total,
                            "splits": {SPLIT_LABELS.get(k, k): v for k, v in stats.splits.items() if v},
                        }
                    )
                except Exception as exc:
                    items.append({"path": path.as_posix(), "name": path.stem, "error": str(exc)})
        return Response.json({"datasets": items, "default": self.default_dataset})

    def _report(self, query: dict[str, list[str]]) -> Response:
        run_id = query.get("run_id", [""])[0]
        fmt = query.get("format", ["html"])[0].lower()
        if not run_id:
            return Response.error("缺少 run_id")
        if db.get_run(self.cfg.project.db_path, run_id) is None:
            return Response.error("运行记录不存在", 404)
        written = write_report(self.cfg, run_id, formats=[fmt])
        if not written:
            return Response.error(f"不支持的报告格式：{fmt}", 400)
        path = written[0]
        download = f"{run_id}-{path.name}"
        return Response.file(path, download_name=download)

    def _submit_eval(self, body: bytes) -> Response:
        payload = _parse_json(body)
        dataset = str(payload.get("dataset") or self.default_dataset)
        name = str(payload.get("name") or f"dashboard-{datetime.now().strftime('%H%M%S')}")
        splits = payload.get("splits") or None
        max_cases = int(payload.get("max_cases") or 0)
        use_judge = bool(payload.get("use_judge", True))
        config_path = str(payload.get("config") or "")
        cfg = Config.load(self.root / config_path) if config_path else self.cfg

        job = self.jobs.create("eval")

        def worker() -> None:
            try:
                cases = load_cases(self.root / dataset)
                if splits:
                    cases = [c for c in cases if c.split in splits]
                if max_cases:
                    cases = cases[:max_cases]
                options = RunOptions(
                    name=name,
                    dataset=dataset,
                    splits=splits,
                    max_cases=max_cases,
                    use_judge=use_judge,
                    persist=True,
                    progress=lambda i, total, result: self.jobs.update(
                        job.job_id, progress=i / max(1, total), message=f"{i}/{total} {result.case.case_id}"
                    ),
                )
                run = run_evaluation(cfg, cases, options)
                self.jobs.update(
                    job.job_id,
                    status="done",
                    progress=1.0,
                    finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    message=f"完成：{run.record.n_cases} 例，坏例 {len(run.flagged)}",
                    result={"run_id": run.record.run_id, "badcases": len(run.flagged)},
                )
            except Exception as exc:
                self.jobs.update(
                    job.job_id,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                )

        threading.Thread(target=worker, daemon=True).start()
        return Response.json({"job_id": job.job_id, "status": "running"}, status=202)

    # -- 静态文件 --------------------------------------------------------
    def _static(self, relative: str) -> Response:
        relative = relative or "index.html"
        if ".." in relative or relative.startswith("/"):
            return Response.error("非法路径", 400)
        path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
            return Response.error("非法路径", 400)
        if not path.exists() or path.suffix.lower() not in SAFE_STATIC:
            return Response.error(f"静态资源不存在：{relative}", 404)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return Response(status=200, body=path.read_bytes(), content_type=f"{content_type}; charset=utf-8")


# --------------------------------------------------------------------------
# 辅助
# --------------------------------------------------------------------------

def _parse_json(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _headline(metrics: list[dict[str, Any]]) -> dict[str, float]:
    wanted = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "citation_accuracy"]
    out = {}
    for metric in metrics:
        if metric.get("name") in wanted:
            out[metric["name"]] = metric.get("value")
    return out


def _count_badcases(db_path: str, run_id: str) -> int:
    return len(db.get_case_results(db_path, run_id, flagged_only=True, limit=10000))


def _dict_record(record: dict[str, Any]):
    from .types import RunRecord

    return RunRecord(
        run_id=record["run_id"],
        name=record["name"],
        created_at=record["created_at"],
        dataset=record["dataset"],
        config_fingerprint=record.get("config_fingerprint", ""),
        config=record.get("config") or {},
        n_cases=record.get("n_cases", 0),
        metrics=record.get("metrics") or [],
        variant=record.get("variant", ""),
    )


def compare_persisted(cfg: Config, left: dict[str, Any], right: dict[str, Any], left_cases: list[dict], right_cases: list[dict]) -> dict[str, Any]:
    """用数据库里的逐用例结果还原配对比较（无需重新跑评测）。"""
    left_map = {c["case_id"]: c.get("scores", {}) for c in left_cases}
    right_map = {c["case_id"]: c.get("scores", {}) for c in right_cases}
    shared = sorted(set(left_map) & set(right_map))
    specs = {s.name: s for s in all_specs()}
    metric_names = sorted(set().union(*(set(m) for m in left_map.values()), *(set(m) for m in right_map.values())))
    from .stats import compare_series

    deltas = []
    for name in metric_names:
        spec = specs.get(name)
        if spec is None or spec.scope != "case":
            continue
        pairs = [(left_map[c][name], right_map[c][name]) for c in shared if name in left_map[c] and name in right_map[c]]
        if len(pairs) < 2:
            continue
        result = compare_series(
            name,
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            direction=spec.direction,
            alpha=cfg.evaluate.significance_alpha,
            samples=max(500, cfg.evaluate.bootstrap_samples),
            seed=f"{left['run_id']}:{right['run_id']}:{name}",
        )
        deltas.append(result.to_dict())

    def quality(scores: dict[str, float]) -> float:
        keys = ["faithfulness", "answer_relevancy", "answer_completeness", "citation_accuracy", "context_recall"]
        vals = [scores[k] for k in keys if k in scores]
        return sum(vals) / len(vals) if vals else 0.0

    case_diffs = [
        {
            "case_id": cid,
            "question": next((c["question"] for c in left_cases if c["case_id"] == cid), ""),
            "left": round(quality(left_map[cid]), 4),
            "right": round(quality(right_map[cid]), 4),
            "delta": round(quality(right_map[cid]) - quality(left_map[cid]), 4),
        }
        for cid in shared
    ]
    case_diffs.sort(key=lambda item: -abs(item["delta"]))
    improved = [d for d in deltas if d.get("significant") and d.get("better") == "right"]
    degraded = [d for d in deltas if d.get("significant") and d.get("better") == "left"]
    return {
        "deltas": deltas,
        "n_cases": len(shared),
        "improved": improved,
        "degraded": degraded,
        "case_diffs": case_diffs[:20],
        "composite": {"left": round(_composite(left), 4), "right": round(_composite(right), 4)},
        "mode": "paired",
    }


def _composite(record: dict[str, Any]) -> float:
    from .experiments import COMPOSITE_METRICS
    from .stats import mean

    specs = {s.name: s for s in all_specs()}
    values = []
    for metric in record.get("metrics") or []:
        if metric["name"] not in COMPOSITE_METRICS:
            continue
        spec = specs.get(metric["name"])
        value = float(metric["value"])
        values.append(value if spec is None or spec.direction == "higher" else 1.0 - value)
    return mean(values) if values else 0.0
