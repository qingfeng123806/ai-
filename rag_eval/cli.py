"""命令行入口：``python -m rag_eval <command>``。

所有子命令都基于同一套库函数，保证「命令行 / 看板 / 定时任务」三种入口
行为一致；任何子命令失败都返回非零退出码，便于 CI 判定。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import __version__, db
from .config import Config, ConfigError
from .dataset import dataset_stats, load_cases, suite_summary_line, write_cases
from .experiments import run_experiment
from .metrics import DEFAULT_FLAG_RULES, all_specs
from .pipeline import build_index, describe_environment, load_pipeline
from .report import write_report
from .runner import RunOptions, run_evaluation
from .types import SPLIT_LABELS

BANNER = "kb-rag-eval · RAG 自动化评测系统"
DEFAULT_SUITE = "data/testsets/suite.jsonl"


# --------------------------------------------------------------------------
# 输出辅助
# --------------------------------------------------------------------------

def _rule(char: str = "─", width: int = 74) -> str:
    return char * width


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Sequence[str] | None = None) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    align = align or ["l"] * len(headers)

    def fmt(values: Sequence[Any]) -> str:
        cells = []
        for i, value in enumerate(values):
            text = str(value)
            cells.append(text.rjust(widths[i]) if align[i] == "r" else text.ljust(widths[i]))
        return "  ".join(cells).rstrip()

    print(fmt(headers))
    print(_rule("─", sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt(row))


def _metric_rows(metrics: Sequence[dict[str, Any]], limit: int = 0) -> list[list[str]]:
    rows: list[list[str]] = []
    for metric in metrics:
        unit = metric.get("name", "")
        value = float(metric.get("value", 0.0))
        if unit.startswith("latency"):
            shown = f"{value:.1f} ms"
        elif unit == "tokens_per_answer":
            shown = f"{value:.0f}"
        elif unit == "cost_per_query":
            shown = f"${value:.5f}"
        else:
            shown = f"{value * 100:.1f}%"
        ci = f"[{metric.get('ci_low', 0):.2f}, {metric.get('ci_high', 0):.2f}]"
        rows.append([metric.get("group", ""), metric.get("name", ""), shown, ci, str(metric.get("n", ""))])
    return rows[:limit] if limit else rows


def _load_config(path: str) -> Config:
    try:
        return Config.load(path)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        raise SystemExit(2)


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    env = describe_environment(cfg)
    print(f"{BANNER} v{__version__}\n")
    print(f"配置文件      : {args.config}")
    print(f"配置指纹      : {cfg.fingerprint()}")
    print(f"数据库        : {cfg.project.db_path}")
    print(f"语料路径      : {', '.join(cfg.ingest.paths)}")
    print(_rule())
    checks: list[tuple[str, str, str]] = [
        ("BGE 向量模型", "可用" if env["embedding"]["bge_available"] else "不可用 → 降级哈希向量化",
         f"provider={cfg.embedding.provider} model={cfg.embedding.model}"),
        ("向量库", "Chroma 可用" if env["store"]["chroma_available"] else
         ("Milvus 可用" if env["store"]["milvus_available"] else "未安装 → 内存向量库"),
         f"provider={cfg.store.provider}"),
        ("LLM 生成", "已配置 API Key" if env["llm"]["api_key_present"] else "未配置 → 抽取式生成",
         f"provider={cfg.generate.provider} model={cfg.generate.model}"),
    ]
    try:
        from .adapters import ragas_status, deepeval_status

        r = ragas_status(cfg)
        d = deepeval_status(cfg)
        checks.append(("RAGAs", "可用" if r.available else f"不可用（{r.reason}）", r.version or ""))
        checks.append(("DeepEval", "可用" if d.available else f"不可用（{d.reason}）", d.version or ""))
    except Exception as exc:
        checks.append(("RAGAs / DeepEval", f"检测失败（{exc}）", ""))
    try:
        import fastapi  # noqa: F401

        checks.append(("FastAPI", "可用（uvicorn 可启动）", getattr(fastapi, "__version__", "")))
    except Exception:
        checks.append(("FastAPI", "未安装 → 使用标准库 HTTP 服务", ""))
    for name, status, detail in checks:
        print(f"{name:<12}: {status}" + (f"  ({detail})" if detail else ""))

    print(_rule())
    docs_dir = Path(cfg.ingest.paths[0]) if cfg.ingest.paths else Path("data/corpus")
    if docs_dir.is_dir():
        files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in (".md", ".txt", ".html", ".pdf", ".docx")]
        print(f"语料文件      : {len(files)} 个（{docs_dir}）")
    suite = Path(DEFAULT_SUITE)
    if suite.exists():
        cases = load_cases(suite)
        print(f"测试集        : {suite_summary_line(cases)}")
    else:
        core = Path("data/testsets/core.jsonl")
        if core.exists():
            print(f"测试集        : {suite_summary_line(load_cases(core))}（core.jsonl，可先运行 testset 命令合并）")
    print(f"指标注册      : {len(all_specs())} 个；坏例规则 {len(DEFAULT_FLAG_RULES)} 条")
    print(f"最近运行      : {len(db.list_runs(cfg.project.db_path, limit=1000))} 条")
    print("\n环境检查完成。")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    print(f"加载语料并构建索引（切片策略：{cfg.ingest.chunker}，chunk_size={cfg.ingest.chunk_size}）…")
    started = time.perf_counter()
    index = build_index(cfg, force=args.force, verbose=True)
    elapsed = time.perf_counter() - started
    stats = index.stats
    print(_rule())
    _print_table(
        ["项目", "数值"],
        [
            ["文档数", len(index.docs)],
            ["切片数", len(index.chunks)],
            ["平均切片长度", f"{stats.avg_chars:.1f} 字"],
            ["最短 / 最长", f"{stats.min_chars} / {stats.max_chars}"],
            ["向量后端", f"{index.embedder.kind}:{index.embedder.dim}d"],
            ["向量库", index.store.kind],
            ["索引指纹", index.signature],
            ["命中缓存", "是" if index.cached else "否（已重建）"],
            ["耗时", f"{elapsed:.2f}s"],
            ["索引目录", index.path],
        ],
        align=["l", "r"],
    )
    return 0


def cmd_testset(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    out = Path(args.out)
    human_paths: list[str] = []
    data_dir = Path("data/testsets")
    for name in ("core.jsonl", "edge.jsonl", "adversarial.jsonl"):
        path = data_dir / name
        if path.exists():
            human_paths.append(str(path))
    human = load_cases(human_paths) if human_paths else []
    print(f"人工标注用例：{len(human)} 条")

    merged = list(human)
    if args.synth:
        from .synth import SynthOptions, synthesize_adversarial, synthesize_edge_from_corpus, synthesize_from_chunks

        index = build_index(cfg)
        opts = SynthOptions(per_chunk=args.per_chunk, max_cases=args.max_cases, seed=cfg.project.seed)
        synthetic = synthesize_from_chunks(index.chunks, opts)
        if args.synth_edge:
            # 边界问题的「正确回答」往往取决于业务预期，机器很难合成出可信参考答案，
            # 因此默认不生成，只在明确需要扩样本压力测试时开启。
            synthetic += synthesize_edge_from_corpus(
                index.chunks, max_cases=max(4, args.max_cases // 3), seed=cfg.project.seed
            )
        if not args.no_adversarial:
            synthetic += synthesize_adversarial(seed=cfg.project.seed)
        print(f"合成候选用例：{len(synthetic)} 条（source=synthetic，建议人工复核）")

        llm = None
        from .llm import build_llm, has_api_key

        if has_api_key(cfg.generate):
            llm = build_llm(cfg.generate)
        if llm is not None and not args.no_rewrite:
            from .synth import rewrite_with_llm

            print("使用 LLM 改写问法…")
            synthetic = rewrite_with_llm(synthetic, llm, model_note=cfg.generate.model)

        existing_questions = {c.question.strip() for c in human}
        merged += [c for c in synthetic if c.question.strip() not in existing_questions]

    # 按问题文本去重：同一文档同一章节的模板问法只保留一条，避免刷高样本量
    deduped: dict[str, Any] = {}
    for case in merged:
        deduped.setdefault(case.question.strip(), case)
    merged = list(deduped.values())

    write_cases(out, merged)
    stats = dataset_stats(merged)
    print(_rule())
    _print_table(
        ["集合", "用例数"],
        [[SPLIT_LABELS.get(k, k), v] for k, v in stats.splits.items() if v] + [["合计", stats.total]],
        align=["l", "r"],
    )
    print(f"来源分布：{stats.sources}")
    print(f"已写入：{out}")
    return 0


def _resolve_dataset(path: str) -> tuple[list, str]:
    if path == "suite":
        path = DEFAULT_SUITE
    target = Path(path)
    if target.is_dir() or target.exists():
        return load_cases(target), str(target)
    core = Path("data/testsets/core.jsonl")
    if core.exists():
        print(f"未找到 {path}，回退到 {core}")
        return load_cases(core), str(core)
    raise SystemExit(f"测试集不存在：{path}")


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    cases, dataset = _resolve_dataset(args.dataset)
    if args.splits:
        wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
        cases = [c for c in cases if c.split in wanted]
    if args.max_cases:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("过滤后测试集为空")

    print(f"{BANNER}")
    print(f"运行名称：{args.name or '(自动)'} | 数据集：{dataset} | 用例：{len(cases)} | 配置：{args.config}")
    print(_rule())

    last_line = [0]

    def progress(index: int, total: int, result) -> None:
        if args.quiet:
            return
        mark = "BAD" if result.flagged else "ok "
        text = f"[{index:>3}/{total}] {mark} {result.case.case_id:<14} {result.case.split:<11} {result.answer[:56]!r}"
        sys.stdout.write("\r" + " " * last_line[0])
        sys.stdout.write("\r" + text[:160])
        last_line[0] = min(160, len(text))
        sys.stdout.flush()

    started = time.perf_counter()
    run = run_evaluation(
        cfg,
        cases,
        RunOptions(
            name=args.name or "",
            dataset=dataset,
            splits=None,
            max_cases=args.max_cases,
            use_judge=not args.no_judge,
            persist=True,
            progress=progress,
        ),
    )
    if not args.quiet:
        print()
    elapsed = time.perf_counter() - started
    record = run.record
    print(_rule())
    print(f"运行 ID：{record.run_id} | 模式：{record.mode} | 耗时：{elapsed:.2f}s | 坏例：{len(run.flagged)}/{record.n_cases}")
    print(_rule())
    _print_table(["分组", "指标", "数值", "95% CI", "n"], _metric_rows(record.metrics), align=["l", "l", "r", "r", "r"])
    print(_rule())
    print("后端：" + " | ".join(f"{k}={v}" for k, v in record.providers.items()))

    if run.flagged and not args.quiet:
        print(f"\n坏例 Top {min(5, len(run.flagged))}：")
        for item in run.flagged[:5]:
            print(f"  · {item.case.case_id} [{item.case.split}] 触发：{', '.join(item.flags)}")
            print(f"    问：{item.case.question[:70]}")
            print(f"    答：{item.answer[:110]}")

    if args.report:
        written = write_report(cfg, record.run_id)
        print("\n报告已导出：")
        for path in written:
            print(f"  - {path}")
    print(f"\n查看看板：python -m rag_eval serve  →  http://127.0.0.1:8000/#/run/{record.run_id}")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    exp = Config.load_experiment(args.experiment)
    dataset = args.dataset or exp.dataset
    cases, resolved = _resolve_dataset(dataset)
    print(f"{BANNER}")
    print(f"实验：{exp.name} | 变体：{len(exp.variants)} | 数据集：{resolved}（{len(cases)} 例）")
    print(exp.description)
    print(_rule())

    def progress(kind: str, *payload) -> None:
        if args.quiet:
            return
        if kind == "variant_start":
            index, total, name = payload
            print(f"\n[{index}/{total}] 变体 {name}")
        elif kind == "case":
            i, total, name, result = payload
            sys.stdout.write(f"\r   {i}/{total} {result.case.case_id:<14} {'BAD' if result.flagged else 'ok '}")
            sys.stdout.flush()

    outcome = run_experiment(cfg, exp, cases, progress=progress)
    if not args.quiet:
        print()
    print(_rule())
    summary = outcome.summary
    print(f"实验 ID：{outcome.experiment_id}")
    print(f"冠军：{summary.get('winner')}（综合分最高）")
    print(_rule())
    rows = []
    for row in summary.get("rows", []):
        if "error" in row:
            rows.append([row["variant"], "失败", row["error"][:40], "", ""])
            continue
        headline = row.get("headline", {})
        rows.append(
            [
                row["variant"],
                f"{row.get('composite', 0):.4f}",
                f"{headline.get('faithfulness', 0) * 100:.0f}% / {headline.get('answer_relevancy', 0) * 100:.0f}%",
                f"{headline.get('context_recall', 0) * 100:.0f}%",
                f"{row.get('duration_s', 0):.2f}s",
            ]
        )
    _print_table(["变体", "综合分", "忠实/相关性", "上下文召回", "耗时"], rows, align=["l", "r", "r", "r", "r"])

    for comparison in outcome.comparisons:
        print(f"\n与基线对比：{comparison['variant']}")
        significant = [d for d in comparison.get("deltas", []) if d.get("significant")]
        if not significant:
            print("  无统计显著差异（可增加用例数或放大小差异）")
            continue
        for delta in significant:
            arrow = "↑" if delta["delta"] > 0 else "↓"
            flag = "改善" if delta["better"] == "right" else "退化"
            print(
                f"  {flag} {delta['metric']:<24} {delta['left']:.3f} → {delta['right']:.3f} "
                f"{arrow}{abs(delta['delta']):.3f}  p={delta['p_value']:.4f}  d={delta['effect_size']:.2f}"
            )
    print(f"\n实验报告：{outcome.report_path}")
    print(f"查看看板：python -m rag_eval serve  →  http://127.0.0.1:8000/#/experiment/{outcome.experiment_id}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    from .api import compare_persisted

    left = db.get_run(cfg.project.db_path, args.left)
    right = db.get_run(cfg.project.db_path, args.right)
    if left is None or right is None:
        raise SystemExit("运行记录不存在，请先执行 runs 查看可用 run_id")
    left_cases = db.get_case_results(cfg.project.db_path, args.left)
    right_cases = db.get_case_results(cfg.project.db_path, args.right)
    comparison = compare_persisted(cfg, left, right, left_cases, right_cases)
    print(f"{BANNER}")
    print(f"基线：{left['name']} ({left['run_id']})")
    print(f"实验：{right['name']} ({right['run_id']})")
    print(f"配对用例：{comparison['n_cases']} | 综合分 {comparison['composite']['left']} → {comparison['composite']['right']}")
    print(_rule())
    rows = []
    for delta in sorted(comparison["deltas"], key=lambda d: (not d.get("significant"), d["metric"])):
        sig = "显著" if delta.get("significant") else ("—" if delta.get("significant") is None else "不显著")
        rows.append(
            [
                delta["metric"],
                f"{delta['left'] * 100:.1f}%",
                f"{delta['right'] * 100:.1f}%",
                f"{delta['delta'] * 100:+.1f}%",
                "—" if delta.get("p_value") is None else f"{delta['p_value']:.4f}",
                sig,
            ]
        )
    _print_table(["指标", "基线", "实验组", "Δ", "p 值", "结论"], rows, align=["l", "r", "r", "r", "r", "l"])
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    run_id = args.run
    if run_id in ("latest", "") or args.latest:
        runs = db.list_runs(cfg.project.db_path, limit=1)
        if not runs:
            raise SystemExit("没有任何运行记录")
        run_id = runs[0]["run_id"]
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    written = write_report(cfg, run_id, formats=formats, on_warning=lambda msg: print(f"警告：{msg}", file=sys.stderr))
    print(f"已导出 {len(written)} 个文件：")
    for path in written:
        print(f"  - {path}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    runs = db.list_runs(cfg.project.db_path, limit=args.limit)
    if not runs:
        print("暂无运行记录。先执行：python -m rag_eval eval --dataset data/testsets/core.jsonl")
        return 0
    rows = []
    for run in runs:
        headline = run.get("metrics") or []
        value = {m["name"]: m["value"] for m in headline}
        bad = len(db.get_case_results(cfg.project.db_path, run["run_id"], flagged_only=True, limit=9999))
        rows.append(
            [
                run["run_id"],
                run["name"][:24],
                run.get("variant", ""),
                str(run.get("n_cases", "")),
                str(bad),
                f"{value.get('faithfulness', 0) * 100:.0f}%",
                f"{value.get('answer_relevancy', 0) * 100:.0f}%",
                run.get("mode", ""),
                run.get("created_at", "")[5:16].replace("T", " "),
            ]
        )
    _print_table(
        ["run_id", "名称", "变体", "用例", "坏例", "忠实", "相关", "模式", "时间"],
        rows,
        align=["l", "l", "l", "r", "r", "r", "r", "l", "l"],
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    framework = args.framework
    if framework == "auto":
        try:
            import uvicorn  # noqa: F401
            import fastapi  # noqa: F401

            framework = "fastapi"
        except Exception:
            framework = "stdlib"
    if framework == "fastapi":
        try:
            import uvicorn

            from .fastapi_app import create_app

            print(f"使用 FastAPI 启动：http://{args.host}:{args.port}")
            uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="warning")
            return 0
        except Exception as exc:
            print(f"FastAPI 启动失败（{exc}），回退到标准库服务器")
    from .dashboard import serve

    serve(cfg, host=args.host, port=args.port, quiet=not args.verbose)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    print(f"{BANNER} · 端到端演示\n")
    print("1) 环境检查")
    env = describe_environment(cfg)
    print(f"   向量化：{env['embedding']['configured']} | 向量库：{env['store']['configured']} | 生成：{env['llm']['configured']}")
    print("\n2) 构建索引")
    index = build_index(cfg, force=args.force, verbose=True)
    print(f"   {len(index.docs)} 篇文档 → {len(index.chunks)} 个切片，后端 {index.embedder.kind}/{index.store.kind}")

    suite = Path(DEFAULT_SUITE)
    if not suite.exists() or args.rebuild_suite:
        print("\n3) 合并测试集")
        cases = load_cases([p for p in ("data/testsets/core.jsonl", "data/testsets/edge.jsonl", "data/testsets/adversarial.jsonl") if Path(p).exists()])
        write_cases(suite, cases)
        print(f"   {suite_summary_line(cases)} → {suite}")
    cases = load_cases(suite)

    print("\n4) 运行全量评测")
    run = run_evaluation(cfg, cases, RunOptions(name="demo-baseline", dataset=str(suite), persist=True))
    print(f"   run_id={run.record.run_id} 模式={run.record.mode} 坏例={len(run.flagged)}/{len(cases)}")
    for metric in run.record.metrics[:6]:
        print(f"     {metric['name']:<24}{metric['value']:.4f}")

    print("\n5) 导出报告")
    written = write_report(cfg, run.record.run_id)
    for path in written:
        print(f"   - {path}")

    print("\n6) 运行 A/B 实验（切片策略）")
    exp = Config.load_experiment("configs/experiments/ab_chunking.toml")
    outcome = run_experiment(cfg, exp, cases, persist=True)
    print(f"   冠军：{outcome.summary.get('winner')}")
    for row in outcome.summary.get("rows", []):
        print(f"     {row['variant']:<18}{row.get('composite', 0):.4f}")

    print("\n完成。启动看板：python -m rag_eval serve")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """无外部依赖的自检：配置、切片、检索、生成、指标、报告全链路各跑一遍。"""
    from .selftest import run_selftest

    return run_selftest(args.config, verbose=not args.quiet)


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # 公共参数既接受「写在子命令前」也接受「写在子命令后」：
    # 用 SUPPRESS 作为默认值，避免子解析器的默认值覆盖用户写在前面传入的值。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="配置文件路径（TOML/JSON）")
    common.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS, help="减少输出")

    parser = argparse.ArgumentParser(
        prog="rag_eval",
        description="面向知识库问答的 RAG 自动化评测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
        epilog="""示例：
  python -m rag_eval doctor
  python -m rag_eval ingest --config configs/baseline.toml
  python -m rag_eval eval --dataset data/testsets/core.jsonl --name baseline
  python -m rag_eval ab --experiment configs/experiments/ab_retrieval.toml
  python -m rag_eval report --run latest --format html,csv,xlsx
  python -m rag_eval serve --port 8000
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="检查运行环境与依赖可用性", parents=[common])
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("ingest", help="加载语料、切片、向量化并构建索引", parents=[common])
    p.add_argument("--force", action="store_true", help="忽略缓存强制重建")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("testset", help="合并/合成测试集", parents=[common])
    p.add_argument("--out", default=DEFAULT_SUITE, help="输出路径")
    p.add_argument("--synth", action="store_true", help="从语料合成候选用例")
    p.add_argument("--per-chunk", type=int, default=1, help="每个切片合成几条")
    p.add_argument("--max-cases", type=int, default=40, help="合成用例上限")
    p.add_argument("--no-adversarial", action="store_true", help="不生成对抗样本")
    p.add_argument("--synth-edge", action="store_true", help="额外合成边界问题（参考答案可信度低，需人工改写）")
    p.add_argument("--no-rewrite", action="store_true", help="即使有 API Key 也不改写问法")
    p.set_defaults(func=cmd_testset)

    p = sub.add_parser("eval", help="运行一次评测", parents=[common])
    p.add_argument("--dataset", default=DEFAULT_SUITE, help="测试集路径（文件或目录）")
    p.add_argument("--name", default="", help="运行名称")
    p.add_argument("--splits", default="", help="只跑某些集合，如 core,edge")
    p.add_argument("--max-cases", type=int, default=0, help="最多用例数")
    p.add_argument("--no-judge", action="store_true", help="关闭 LLM-as-Judge")
    p.add_argument("--no-report", dest="report", action="store_false", help="不自动导出报告")
    p.set_defaults(func=cmd_eval, report=True)

    p = sub.add_parser("ab", help="运行 A/B 实验", parents=[common])
    p.add_argument("--experiment", required=True, help="实验配置路径")
    p.add_argument("--dataset", default="", help="覆盖实验配置中的测试集")
    p.set_defaults(func=cmd_ab)

    p = sub.add_parser("compare", help="对比两次运行", parents=[common])
    p.add_argument("--left", required=True, help="基线 run_id")
    p.add_argument("--right", required=True, help="实验组 run_id")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report", help="导出评测报告", parents=[common])
    p.add_argument("--run", default="latest", help="run_id，或 latest")
    p.add_argument("--latest", action="store_true", help="等价于 --run latest")
    p.add_argument("--format", default="html,markdown,csv,json", help="逗号分隔：html,markdown,csv,json,xlsx")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("runs", help="列出历史运行", parents=[common])
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("serve", help="启动评测看板与 API", parents=[common])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--framework", choices=["auto", "stdlib", "fastapi"], default="auto")
    p.add_argument("--verbose", action="store_true", help="打印访问日志")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("demo", help="端到端演示：建索引→评测→报告→A/B", parents=[common])
    p.add_argument("--force", action="store_true", help="强制重建索引")
    p.add_argument("--rebuild-suite", action="store_true", help="重新合并测试集")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("selftest", help="离线自检（不依赖第三方包）", parents=[common])
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.config = getattr(args, "config", None) or "configs/baseline.toml"
    args.quiet = bool(getattr(args, "quiet", False))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
