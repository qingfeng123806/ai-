"""A/B 实验与数据库测试。"""

from rag_eval import db
from rag_eval.config import Config, ExperimentCfg
from rag_eval.dataset import load_cases
from rag_eval.experiments import composite_score, run_experiment
from rag_eval.runner import RunOptions, run_evaluation
from tests.helpers import temp_config, tempdir

_CACHE: dict = {}


def _experiment():
    if "outcome" not in _CACHE:
        tmp = tempdir()
        cfg = temp_config(tmp, evaluate={"bootstrap_samples": 200})
        exp = ExperimentCfg.from_dict(
            {
                "name": "unit-ab",
                "dataset": "data/testsets/core.jsonl",
                "variants": [
                    {"name": "top3", "overrides": {"retrieve": {"top_k": 3, "candidate_k": 6}}},
                    {"name": "top6", "overrides": {"retrieve": {"top_k": 6, "candidate_k": 12}}},
                ],
            }
        )
        cases = load_cases("data/testsets/core.jsonl")[:5]
        outcome = run_experiment(cfg, exp, cases, persist=True)
        _CACHE.update({"cfg": cfg, "outcome": outcome, "tmp": tmp, "exp": exp})
    return _CACHE["cfg"], _CACHE["outcome"]


def test_experiment_runs_all_variants():
    cfg, outcome = _experiment()
    assert len(outcome.outcomes) == 2
    assert all(item.run is not None for item in outcome.outcomes), [o.error for o in outcome.outcomes]
    assert outcome.summary["winner"] in {"top3", "top6"}
    assert outcome.summary["n_variants"] == 2
    assert len(outcome.summary["ranking"]) == 2


def test_experiment_comparisons_present():
    cfg, outcome = _experiment()
    assert len(outcome.comparisons) == 1
    comparison = outcome.comparisons[0]
    assert comparison["variant"] == "top6"
    assert comparison["n_cases"] == 5
    assert comparison["deltas"]
    assert all("p_value" in delta for delta in comparison["deltas"])
    assert comparison["composite"]["left"] > 0


def test_composite_score_direction_aware():
    cfg, outcome = _experiment()
    run = outcome.outcomes[0].run
    score = composite_score(run.record)
    assert 0.0 <= score <= 1.0


def test_experiment_persisted_and_listed():
    cfg, outcome = _experiment()
    experiments = db.list_experiments(cfg.project.db_path)
    assert any(item["experiment_id"] == outcome.experiment_id for item in experiments)
    detail = db.get_experiment(cfg.project.db_path, outcome.experiment_id)
    assert detail is not None
    assert detail["runs"]
    assert detail["summary"]["winner"]


def test_experiment_artifact_written():
    cfg, outcome = _experiment()
    from pathlib import Path

    assert outcome.report_path
    assert Path(outcome.report_path).exists()


def test_db_delete_run():
    tmp = tempdir()
    cfg = temp_config(tmp)
    run = run_evaluation(cfg, load_cases("data/testsets/core.jsonl")[:1], RunOptions(persist=True))
    assert db.get_run(cfg.project.db_path, run.record.run_id) is not None
    deleted = db.delete_run(cfg.project.db_path, run.record.run_id)
    assert deleted == 1
    assert db.get_run(cfg.project.db_path, run.record.run_id) is None
    assert db.get_case_results(cfg.project.db_path, run.record.run_id) == []


def test_override_does_not_mutate_base_config():
    cfg = temp_config(tempdir())
    before = cfg.to_dict()
    cfg.apply_overrides({"retrieve": {"top_k": 1}})
    assert cfg.to_dict() == before


def test_experiment_config_requires_dataset():
    try:
        ExperimentCfg.from_dict({"name": "x", "variants": [{"name": "a"}, {"name": "b"}]})
    except Exception:
        pass
    else:
        raise AssertionError("缺少 dataset 应当报错")


def test_config_from_json_roundtrip():
    cfg = Config.from_dict({"retrieve": {"top_k": 2, "candidate_k": 4}})
    assert cfg.retrieve.top_k == 2
    again = Config.from_dict(cfg.to_dict(), "inline")
    assert again.fingerprint() == cfg.fingerprint()
