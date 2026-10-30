"""运行器测试：端到端评测、持久化与指标选筛。"""

import json
from pathlib import Path

from rag_eval import db
from rag_eval.dataset import load_cases
from rag_eval.runner import RunOptions, resolve_metric_names, run_evaluation
from tests.helpers import temp_config, tempdir, make_case


def test_resolve_metric_names_filters_judge():
    cfg = temp_config(tempdir(), evaluate={"metrics": ["faithfulness", "judge_overall", "latency_p95"]})
    with_judge = resolve_metric_names(cfg, use_judge=True)
    without = resolve_metric_names(cfg, use_judge=False)
    assert "judge_overall" in with_judge
    assert "judge_overall" not in without
    assert "faithfulness" in without and "latency_p95" in without


def test_run_evaluation_persists_and_scores():
    tmp = tempdir()
    cfg = temp_config(tmp)
    cases = [make_case(question="团队版价格是多少？")]
    run = run_evaluation(cfg, cases, RunOptions(name="unit-run", dataset="unit", persist=True))
    record = run.record
    assert record.n_cases == 1
    assert record.metrics
    assert record.mode in ("offline", "hybrid", "live")
    assert run.results[0].scores.get("faithfulness") is not None
    assert run.results[0].answer

    stored = db.get_run(cfg.project.db_path, record.run_id)
    assert stored is not None and stored["name"] == "unit-run"
    cases_in_db = db.get_case_results(cfg.project.db_path, record.run_id)
    assert len(cases_in_db) == 1
    assert cases_in_db[0]["case_id"] == "t-001"

    artifact = Path(cfg.project.artifacts_dir) / "runs" / f"{record.run_id}.jsonl"
    assert artifact.exists()
    lines = [line for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert "faithfulness" in json.loads(lines[0])["scores"]


def test_run_without_persistence():
    tmp = tempdir()
    cfg = temp_config(tmp)
    run = run_evaluation(cfg, [make_case()], RunOptions(persist=False))
    assert run.record.run_id
    assert db.get_run(cfg.project.db_path, run.record.run_id) is None


def test_run_flags_badcases_on_real_corpus():
    tmp = tempdir()
    cfg = temp_config(tmp)
    cases = load_cases("data/testsets/core.jsonl")[:4]
    run = run_evaluation(cfg, cases, RunOptions(persist=False))
    assert all(isinstance(item.flagged, bool) for item in run.results)
    assert any(item.scores for item in run.results)


def test_empty_dataset_rejected():
    tmp = tempdir()
    cfg = temp_config(tmp)
    try:
        run_evaluation(cfg, [], RunOptions(persist=False))
    except ValueError:
        pass
    else:
        raise AssertionError("空测试集应当报错")


def test_judge_scores_recorded_in_details():
    tmp = tempdir()
    cfg = temp_config(tmp)
    run = run_evaluation(cfg, [make_case()], RunOptions(persist=False, use_judge=True))
    judge = run.results[0].details.get("judge")
    assert judge is not None
    assert judge["source"] in ("heuristic-judge", "llm-judge")
    assert "faithfulness" in judge["scores"]
