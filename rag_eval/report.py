"""报告导出：自包含 HTML、Markdown、CSV、JSON、XLSX。

HTML 报告不依赖任何 CDN 或前端构建产物，单文件即可离线打开与归档，
适合作为「每次实验的存档物」；看板（dashboard）则是可交互的实时视图。
两者共用同一份 report payload，避免口径不一致。
"""

from __future__ import annotations

import csv
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import db
from .config import Config
from .metrics import specs_by_name
from .text import truncate
from .types import SPLIT_LABELS

GROUP_ORDER = ["retrieval", "generation", "citation", "judge", "safety", "performance", "external"]
GROUP_LABELS = {
    "retrieval": "检索质量",
    "generation": "生成质量",
    "citation": "引用质量",
    "judge": "LLM-as-Judge",
    "safety": "安全合规",
    "performance": "性能与成本",
    "external": "外部框架复算",
}
HEADLINE = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "hallucination_rate",
    "citation_accuracy",
]


def _fmt(metric: dict[str, Any]) -> str:
    name = metric["name"]
    value = float(metric["value"])
    unit = (specs_by_name().get(name).unit if specs_by_name().get(name) else "score")
    if name.startswith("latency"):
        return f"{value:.1f} ms"
    if unit == "tokens":
        return f"{value:.1f}"
    if unit == "usd":
        return f"${value:.5f}"
    return f"{value * 100:.1f}%"


def collect(cfg: Config, run_id: str, max_badcases: int = 30) -> dict[str, Any]:
    record = db.get_run(cfg.project.db_path, run_id)
    if record is None:
        raise ValueError(f"运行记录不存在：{run_id}")
    cases = db.get_case_results(cfg.project.db_path, run_id)
    metrics = record.get("metrics") or []
    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUP_ORDER}
    for metric in metrics:
        grouped.setdefault(metric.get("group", "other"), []).append(metric)
    for group in grouped:
        grouped[group].sort(key=lambda m: (m.get("group") != group, m["name"]))
    badcases = [c for c in cases if c.get("flagged")]
    # 按 core → edge → adversarial 的固定顺序输出，避免依赖数据库返回顺序
    splits: dict[str, dict[str, int]] = {}
    for split in SPLIT_LABELS:
        bucket = {"n": 0, "bad": 0}
        for case in cases:
            if case.get("split") == split:
                bucket["n"] += 1
                if case.get("flagged"):
                    bucket["bad"] += 1
        if bucket["n"]:
            splits[split] = bucket
    for case in cases:
        split = case.get("split", "core")
        if split not in splits:
            splits[split] = {"n": 0, "bad": 0}
            for item in cases:
                if item.get("split") == split:
                    splits[split]["n"] += 1
                    if item.get("flagged"):
                        splits[split]["bad"] += 1
    return {
        "title": cfg.report.title,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run": record,
        "metrics": grouped,
        "metric_list": metrics,
        "cases": cases,
        "badcases": badcases[:max_badcases],
        "badcase_total": len(badcases),
        "splits": splits,
        "split_labels": SPLIT_LABELS,
        "providers": record.get("providers") or {},
        "mode": record.get("mode", "offline"),
        "scorecard": _scorecard(metrics),
    }


def _scorecard(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {m["name"]: m for m in metrics}
    out: list[dict[str, Any]] = []
    for name in HEADLINE:
        metric = by_name.get(name)
        if metric is None:
            continue
        out.append(
            {
                "name": name,
                "label": _METRIC_LABELS.get(name, name),
                "value": float(metric["value"]),
                "display": _fmt(metric),
                "direction": metric.get("direction", "higher"),
                "ci": [metric.get("ci_low"), metric.get("ci_high")],
            }
        )
    return out


_METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "hallucination_rate": "幻觉率",
    "citation_accuracy": "引用准确率",
    "hit_rate": "命中率",
    "mrr": "MRR",
    "ndcg": "nDCG",
    "answer_completeness": "答案完整性",
}


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def render_markdown(payload: dict[str, Any]) -> str:
    run = payload["run"]
    lines: list[str] = []
    lines.append(f"# {payload['title']}")
    lines.append("")
    lines.append(f"- 运行 ID：`{run['run_id']}`")
    lines.append(f"- 运行名称：{run['name']}")
    lines.append(f"- 生成时间：{payload['generated_at']}")
    lines.append(f"- 数据集：`{run['dataset']}`（{run['n_cases']} 例）")
    lines.append(f"- 运行模式：{payload['mode']}")
    lines.append(f"- 配置指纹：`{run['config_fingerprint']}`")
    providers = payload["providers"]
    if providers:
        lines.append("- 实际后端：")
        for key, value in providers.items():
            lines.append(f"  - {key}: `{value}`")
    lines.append("")

    lines.append("## 指标总览")
    lines.append("")
    lines.append("| 分组 | 指标 | 数值 | 95% 置信区间 | 样本 | 方向 |")
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    for group in GROUP_ORDER:
        for metric in payload["metrics"].get(group, []):
            ci = f"[{metric['ci_low'] * 100:.1f}, {metric['ci_high'] * 100:.1f}]"
            arrow = "越高越好" if metric.get("direction") == "higher" else "越低越好"
            lines.append(
                f"| {GROUP_LABELS.get(group, group)} | {metric['name']} | {_fmt(metric)} | {ci} | {metric.get('n', 0)} | {arrow} |"
            )
    lines.append("")

    lines.append("## 分集合表现")
    lines.append("")
    lines.append("| 集合 | 用例数 | 坏例数 | 坏例率 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for split, bucket in payload["splits"].items():
        rate = bucket["bad"] / bucket["n"] if bucket["n"] else 0.0
        lines.append(f"| {SPLIT_LABELS.get(split, split)} | {bucket['n']} | {bucket['bad']} | {rate * 100:.1f}% |")
    lines.append("")

    lines.append(f"## 坏例清单（共 {payload['badcase_total']} 条，展示前 {len(payload['badcases'])} 条）")
    lines.append("")
    for case in payload["badcases"]:
        lines.append(f"### {case['case_id']} · {SPLIT_LABELS.get(case['split'], case['split'])}")
        lines.append("")
        lines.append(f"- 问题：{case['question']}")
        lines.append(f"- 触发指标：{', '.join(case.get('flags') or []) or '无'}")
        lines.append(f"- 答案：{truncate(case.get('answer', ''), 400)}")
        ref = (case.get("case") or {}).get("reference_answer")
        if ref:
            lines.append(f"- 参考答案：{truncate(ref, 300)}")
        score_bits = [f"{k}={v:.2f}" for k, v in (case.get("scores") or {}).items() if k in HEADLINE]
        if score_bits:
            lines.append(f"- 关键分数：{', '.join(score_bits)}")
        judge = (case.get("details") or {}).get("judge") or {}
        reason = (judge.get("reasons") or {}).get("overall")
        if reason:
            lines.append(f"- Judge 意见：{reason}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------

def render_metrics_csv(payload: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["group", "metric", "value", "ci_low", "ci_high", "n", "direction", "description"])
    for group in GROUP_ORDER:
        for metric in payload["metrics"].get(group, []):
            writer.writerow(
                [
                    group,
                    metric["name"],
                    metric["value"],
                    metric.get("ci_low"),
                    metric.get("ci_high"),
                    metric.get("n"),
                    metric.get("direction"),
                    metric.get("description", ""),
                ]
            )
    return buffer.getvalue()


def render_cases_csv(payload: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    metrics = [m["name"] for m in payload["metric_list"] if m.get("scope") == "case"]
    writer.writerow(
        ["case_id", "split", "question", "answer", "refused", "flagged", "flags", "latency_ms", "error"]
        + metrics
    )
    for case in payload["cases"]:
        writer.writerow(
            [
                case["case_id"],
                case["split"],
                case["question"],
                case.get("answer", ""),
                case.get("refused"),
                case.get("flagged"),
                ";".join(case.get("flags") or []),
                (case.get("latency") or {}).get("total_ms"),
                case.get("error", ""),
            ]
            + [case.get("scores", {}).get(name, "") for name in metrics]
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------

def write_xlsx(payload: dict[str, Any], path: str | Path) -> Path:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ModuleNotFoundError as exc:
        raise RuntimeError("导出 XLSX 需要 openpyxl：pip install openpyxl") from exc

    wb = Workbook()
    head_fill = PatternFill("solid", fgColor="1F2A44")
    head_font = Font(color="FFFFFF", bold=True)

    def style_header(ws) -> None:
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(vertical="center")
        ws.freeze_panes = "A2"

    ws = wb.active
    ws.title = "概览"
    run = payload["run"]
    rows = [
        ("运行 ID", run["run_id"]),
        ("运行名称", run["name"]),
        ("生成时间", payload["generated_at"]),
        ("数据集", run["dataset"]),
        ("用例数", run["n_cases"]),
        ("运行模式", payload["mode"]),
        ("配置指纹", run["config_fingerprint"]),
        ("坏例数", payload["badcase_total"]),
        ("耗时(秒)", run.get("duration_s")),
    ]
    ws.append(["字段", "值"])
    for row in rows:
        ws.append(list(row))
    style_header(ws)
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 60

    ws = wb.create_sheet("指标")
    ws.append(["分组", "指标", "数值", "CI 下限", "CI 上限", "样本数", "方向", "说明"])
    for group in GROUP_ORDER:
        for metric in payload["metrics"].get(group, []):
            ws.append(
                [
                    GROUP_LABELS.get(group, group),
                    metric["name"],
                    metric["value"],
                    metric.get("ci_low"),
                    metric.get("ci_high"),
                    metric.get("n"),
                    "higher" if metric.get("direction") == "higher" else "lower",
                    metric.get("description", ""),
                ]
            )
    style_header(ws)
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["H"].width = 60

    ws = wb.create_sheet("用例结果")
    metric_names = [m["name"] for m in payload["metric_list"] if m.get("scope") == "case"]
    ws.append(["case_id", "split", "question", "answer", "flagged", "flags", "latency_ms", "error"] + metric_names)
    for case in payload["cases"]:
        ws.append(
            [
                case["case_id"],
                case["split"],
                case["question"],
                case.get("answer", ""),
                case.get("flagged"),
                ";".join(case.get("flags") or []),
                (case.get("latency") or {}).get("total_ms"),
                case.get("error", ""),
            ]
            + [case.get("scores", {}).get(name) for name in metric_names]
        )
    style_header(ws)
    ws.column_dimensions["C"].width = 46
    ws.column_dimensions["D"].width = 60

    ws = wb.create_sheet("坏例")
    ws.append(["case_id", "split", "问题", "触发指标", "答案", "参考答案", "Judge 意见"])
    for case in payload["badcases"]:
        judge = (case.get("details") or {}).get("judge") or {}
        ws.append(
            [
                case["case_id"],
                SPLIT_LABELS.get(case["split"], case["split"]),
                case["question"],
                ", ".join(case.get("flags") or []),
                case.get("answer", ""),
                (case.get("case") or {}).get("reference_answer", ""),
                (judge.get("reasons") or {}).get("overall", ""),
            ]
        )
    style_header(ws)
    for col, width in (("C", 46), ("D", 28), ("E", 60), ("F", 50), ("G", 40)):
        ws.column_dimensions[col].width = width

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_CSS = """
:root{--bg:#0f1420;--panel:#161d2d;--panel2:#1c2436;--line:#263148;--text:#e6ebf5;
--muted:#8fa1c0;--good:#3ddc97;--bad:#ff6b6b;--warn:#ffc857;--accent:#5aa9ff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.65 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 64px}
header{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:.5px}
h2{font-size:17px;margin:34px 0 14px;padding-left:10px;border-left:3px solid var(--accent)}
h3{font-size:14px;margin:0 0 8px}
.sub{color:var(--muted);font-size:13px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.tag{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:3px 9px;font-size:12px;color:var(--muted)}
.tag b{color:var(--text);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--muted);font-size:12px}
.card .v{font-size:24px;font-weight:600;margin-top:6px;font-variant-numeric:tabular-nums}
.bar{height:6px;border-radius:3px;background:var(--panel2);margin-top:10px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#5aa9ff,#3ddc97)}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{background:var(--panel2);color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
tr:last-child td{border-bottom:none}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;border:1px solid var(--line)}
.pill.ok{color:var(--good);border-color:rgba(61,220,151,.4);background:rgba(61,220,151,.08)}
.pill.bad{color:var(--bad);border-color:rgba(255,107,107,.4);background:rgba(255,107,107,.08)}
.case{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:14px}
.case .q{font-weight:600;margin-bottom:8px}
.case .row{margin:6px 0}
.lbl{color:var(--muted);font-size:12px;margin-right:6px}
pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px;margin:6px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
.ctx{border-left:2px solid var(--accent);padding-left:10px;margin:8px 0;color:#cfd9ea}
.ctx .h{font-size:12px;color:var(--muted)}
.sup{font-size:12px;color:var(--muted)}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""


def _bar(value: float, direction: str = "higher") -> str:
    pct = max(0.0, min(1.0, value))
    return f'<div class="bar"><i style="width:{pct * 100:.1f}%"></i></div>'


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def render_html(payload: dict[str, Any]) -> str:
    run = payload["run"]
    providers = payload["providers"]
    cards: list[str] = []
    for item in payload["scorecard"]:
        cards.append(
            f'<div class="card"><div class="k">{_esc(item["label"])} · {_esc(item["name"])}</div>'
            f'<div class="v">{_esc(item["display"])}</div>{_bar(item["value"])}</div>'
        )
    tag_html = "".join(f'<span class="tag">{_esc(k)} <b>{_esc(v)}</b></span>' for k, v in providers.items())

    metric_rows: list[str] = []
    for group in GROUP_ORDER:
        items = payload["metrics"].get(group, [])
        if not items:
            continue
        metric_rows.append(f'<tr><th colspan="6" style="color:var(--accent)">{_esc(GROUP_LABELS.get(group, group))}</th></tr>')
        for metric in items:
            ci = f"{metrics_ci(metric)}"
            metric_rows.append(
                f"<tr><td>{_esc(metric['name'])}</td><td>{_esc(metric.get('description', ''))}</td>"
                f'<td class="num">{_esc(_fmt(metric))}</td><td class="num">{_esc(ci)}</td>'
                f'<td class="num">{_esc(metric.get("n", 0))}</td>'
                f'<td>{"越高越好" if metric.get("direction") == "higher" else "越低越好"}</td></tr>'
            )

    split_rows = "".join(
        f"<tr><td>{_esc(SPLIT_LABELS.get(split, split))}</td><td class='num'>{bucket['n']}</td>"
        f"<td class='num'>{bucket['bad']}</td>"
        f"<td class='num'>{bucket['bad'] / bucket['n'] * 100:.1f}%</td></tr>"
        for split, bucket in payload["splits"].items()
    )

    badcase_html: list[str] = []
    for case in payload["badcases"]:
        scores = "".join(
            f'<span class="tag">{_esc(k)} <b>{v:.2f}</b></span>'
            for k, v in (case.get("scores") or {}).items()
            if k in HEADLINE
        )
        flags = "".join(f'<span class="pill bad">{_esc(f)}</span> ' for f in (case.get("flags") or []))
        ctx_blocks = []
        for ctx_item in (case.get("contexts") or [])[:4]:
            ctx_blocks.append(
                f'<div class="ctx"><div class="h">[{_esc(ctx_item.get("rank"))}] '
                f'{_esc(ctx_item.get("title"))} · score {_esc(ctx_item.get("score"))} · '
                f'relevant {"是" if _is_relevant(case, ctx_item) else "否"}</div>'
                f'{_esc(truncate(ctx_item.get("text", ""), 320))}</div>'
            )
        support_rows = []
        for sup in (case.get("details") or {}).get("support", [])[:6]:
            mark = "支撑" if sup.get("supported") else "未支撑"
            support_rows.append(
                f"<div class='sup'>[{mark}] {_esc(truncate(sup.get('sentence', ''), 160))}"
                f" · support={_esc(sup.get('support'))}</div>"
            )
        judge = (case.get("details") or {}).get("judge") or {}
        reason = (judge.get("reasons") or {}).get("overall") or ""
        ref = (case.get("case") or {}).get("reference_answer") or ""
        badcase_html.append(
            f"""<div class="case">
  <div class="q">{_esc(case["case_id"])} · {_esc(SPLIT_LABELS.get(case["split"], case["split"]))} · {flags}</div>
  <div class="row"><span class="lbl">问题</span>{_esc(case["question"])}</div>
  <div class="row"><span class="lbl">答案</span></div>
  <pre>{_esc(case.get("answer", ""))}</pre>
  <div class="row"><span class="lbl">参考答案</span>{_esc(ref) or "—"}</div>
  <div class="row">{scores}</div>
  {f'<div class="row"><span class="lbl">Judge</span>{_esc(reason)}</div>' if reason else ''}
  <details><summary class="sub">逐句支撑度与检索片段</summary>
    {''.join(support_rows)}
    {''.join(ctx_blocks)}
  </details>
</div>"""
        )

    title = _esc(payload["title"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title} · {_esc(run["run_id"])}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>{title}</h1>
  <div class="sub">{_esc(run["name"])} · 生成于 {_esc(payload["generated_at"])}</div>
  <div class="meta">
    <span class="tag">run_id <b>{_esc(run["run_id"])}</b></span>
    <span class="tag">数据集 <b>{_esc(run["dataset"])}</b></span>
    <span class="tag">用例 <b>{_esc(run["n_cases"])}</b></span>
    <span class="tag">坏例 <b>{_esc(payload["badcase_total"])}</b></span>
    <span class="tag">模式 <b>{_esc(payload["mode"])}</b></span>
    <span class="tag">耗时 <b>{_esc(run.get("duration_s", 0))}s</b></span>
  </div>
  <div class="meta">{tag_html}</div>
</header>

<h2>核心指标</h2>
<div class="cards">{''.join(cards)}</div>

<h2>分集合表现</h2>
<table><thead><tr><th>集合</th><th>用例数</th><th>坏例数</th><th>坏例率</th></tr></thead>
<tbody>{split_rows}</tbody></table>

<h2>全部指标</h2>
<table>
<colgroup><col style="width:20%"/><col style="width:40%"/><col style="width:10%"/><col style="width:14%"/><col style="width:6%"/><col style="width:10%"/></colgroup>
<thead><tr><th>指标</th><th>口径</th><th>数值</th><th>95% CI</th><th>n</th><th>方向</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody></table>

<h2>坏例清单（{payload["badcase_total"]} 条，展示前 {len(payload["badcases"])} 条）</h2>
{''.join(badcase_html) or '<p class="sub">没有坏例。</p>'}

<footer>由 kb-rag-eval 生成 · 指标口径见 docs/metrics.md · 配置文件指纹 {_esc(run["config_fingerprint"])}</footer>
</div></body></html>
"""


def metrics_ci(metric: dict[str, Any]) -> str:
    low = metric.get("ci_low")
    high = metric.get("ci_high")
    if low is None or high is None:
        return "—"
    if metric["name"].startswith("latency"):
        return f"[{low:.1f}, {high:.1f}]"
    return f"[{low * 100:.1f}, {high * 100:.1f}]"


def _is_relevant(case: dict[str, Any], ctx_item: dict[str, Any]) -> bool:
    for item in (case.get("details") or {}).get("relevance", []):
        if item.get("chunk_id") == ctx_item.get("chunk_id"):
            return bool(item.get("relevant"))
    return False


# --------------------------------------------------------------------------
# 导出入口
# --------------------------------------------------------------------------

def write_report(
    cfg: Config,
    run_id: str,
    formats: Iterable[str] | None = None,
    out_dir: str | Path | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> list[Path]:
    payload = collect(cfg, run_id, max_badcases=cfg.report.max_badcases)
    target_dir = Path(out_dir or cfg.report.out_dir) / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = [f.lower() for f in (formats or cfg.report.formats)]
    aliases = {"md": "markdown", "htm": "html", "xls": "xlsx"}
    wanted = [aliases.get(w, w) for w in wanted]
    written: list[Path] = []

    if "html" in wanted:
        path = target_dir / "report.html"
        path.write_text(render_html(payload), encoding="utf-8")
        written.append(path)
    if "markdown" in wanted:
        path = target_dir / "report.md"
        path.write_text(render_markdown(payload), encoding="utf-8")
        written.append(path)
    if "json" in wanted:
        path = target_dir / "report.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(path)
    if "csv" in wanted:
        path = target_dir / "metrics.csv"
        path.write_text(render_metrics_csv(payload), encoding="utf-8-sig")
        written.append(path)
        path = target_dir / "cases.csv"
        path.write_text(render_cases_csv(payload), encoding="utf-8-sig")
        written.append(path)
    if "xlsx" in wanted:
        try:
            written.append(write_xlsx(payload, target_dir / "report.xlsx"))
        except RuntimeError as exc:
            # 缺少 openpyxl 时只跳过该格式，其余格式照常导出
            if on_warning is not None:
                on_warning(f"跳过 XLSX：{exc}")

    for path in written:
        try:
            db.register_artifact(cfg.project.db_path, run_id, str(path), path.suffix.lstrip("."), payload["generated_at"])
        except Exception:
            continue
    return written
