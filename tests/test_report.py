"""报告导出测试：HTML / Markdown / CSV / JSON / XLSX。"""

import json
from pathlib import Path

from rag_eval.dataset import load_cases
from rag_eval.report import (
    collect,
    render_cases_csv,
    render_html,
    render_markdown,
    render_metrics_csv,
    write_report,
    write_xlsx,
)
from rag_eval.runner import RunOptions, run_evaluation
from tests.helpers import temp_config, tempdir

_CACHE: dict = {}


def _run():
    if "run" not in _CACHE:
        tmp = tempdir()
        cfg = temp_config(tmp)
        cases = load_cases("data/testsets/core.jsonl")[:2]
        run = run_evaluation(cfg, cases, RunOptions(name="report-run", dataset="core", persist=True))
        _CACHE.update({"cfg": cfg, "run": run, "tmp": tmp})
    return _CACHE["cfg"], _CACHE["run"]


def test_collect_payload_structure():
    cfg, run = _run()
    payload = collect(cfg, run.record.run_id)
    assert payload["run"]["run_id"] == run.record.run_id
    assert payload["metrics"]["retrieval"] or payload["metrics"]["generation"]
    assert payload["scorecard"]
    assert payload["generated_at"]
    assert payload["metric_list"]


def test_markdown_contains_sections():
    cfg, run = _run()
    payload = collect(cfg, run.record.run_id)
    md = render_markdown(payload)
    assert "# " in md
    assert "## 指标总览" in md
    assert "| 分组 | 指标 |" in md
    assert run.record.run_id in md


def test_html_is_self_contained():
    cfg, run = _run()
    html = render_html(collect(cfg, run.record.run_id))
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html, "报告不应依赖外部资源"
    assert "核心指标" in html


def test_csv_export_has_header_and_rows():
    cfg, run = _run()
    payload = collect(cfg, run.record.run_id)
    metrics_csv = render_metrics_csv(payload)
    cases_csv = render_cases_csv(payload)
    assert metrics_csv.splitlines()[0].startswith("group,metric,value")
    assert "faithfulness" in metrics_csv
    assert cases_csv.splitlines()[0].startswith("case_id,split,question")
    assert "t-001" in cases_csv or "core-" in cases_csv


def test_write_report_formats():
    cfg, run = _run()
    out_dir = Path(cfg.project.artifacts_dir) / "test-reports"
    written = write_report(cfg, run.record.run_id, formats=["html", "markdown", "json", "csv"], out_dir=out_dir)
    names = {path.name for path in written}
    assert {"report.html", "report.md", "report.json", "metrics.csv", "cases.csv"} <= names
    payload = json.loads((out_dir / run.record.run_id / "report.json").read_text(encoding="utf-8"))
    assert payload["run"]["run_id"] == run.record.run_id


def test_write_xlsx_optional():
    cfg, run = _run()
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        return  # 无 openpyxl 时跳过（不得让测试失败）
    path = write_xlsx(collect(cfg, run.record.run_id), Path(cfg.project.artifacts_dir) / "test-reports" / "r.xlsx")
    assert path.exists() and path.stat().st_size > 2000


def test_missing_run_raises():
    cfg, _ = _run()
    try:
        collect(cfg, "no-such-run")
    except ValueError:
        pass
    else:
        raise AssertionError("不存在的 run 应当报错")
