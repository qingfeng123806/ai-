"""配置加载、校验、覆盖与指纹测试。"""

from rag_eval.config import Config, ConfigError, ExperimentCfg


def test_defaults_are_valid():
    cfg = Config.from_dict({})
    assert cfg.ingest.chunker == "recursive"
    assert cfg.retrieve.top_k == 6
    assert cfg.fingerprint()


def test_load_baseline_toml():
    cfg = Config.load("configs/baseline.toml")
    assert cfg.project.name == "kb-rag-eval"
    assert "faithfulness" in cfg.evaluate.metrics
    assert cfg.embedding.dim > 0


def test_unknown_section_rejected():
    try:
        Config.from_dict({"pipeline": {}})
    except ConfigError as exc:
        assert "pipeline" in str(exc)
    else:
        raise AssertionError("应当拒绝未知配置段")


def test_invalid_enum_rejected():
    try:
        Config.from_dict({"ingest": {"chunker": "magic"}})
    except ConfigError as exc:
        assert "chunker" in str(exc)
    else:
        raise AssertionError("应当拒绝非法切片策略")


def test_invalid_range_rejected():
    try:
        Config.from_dict({"retrieve": {"top_k": 0}})
    except ConfigError:
        pass
    else:
        raise AssertionError("应当拒绝 top_k=0")

    try:
        Config.from_dict({"retrieve": {"top_k": 10, "candidate_k": 5}})
    except ConfigError:
        pass
    else:
        raise AssertionError("应当拒绝 candidate_k < top_k")


def test_overrides_are_deep_merged_and_immutable():
    cfg = Config.load("configs/baseline.toml")
    changed = cfg.apply_overrides({"retrieve": {"top_k": 12}, "rerank": {"provider": "none"}})
    assert changed.retrieve.top_k == 12
    assert changed.rerank.provider == "none"
    assert cfg.retrieve.top_k == 6, "原配置不应被修改"
    assert changed.fingerprint() != cfg.fingerprint()


def test_overrides_reject_unknown_section():
    cfg = Config.load("configs/baseline.toml")
    try:
        cfg.apply_overrides({"unknown": {"x": 1}})
    except ConfigError:
        pass
    else:
        raise AssertionError("应当拒绝未知覆盖段")


def test_fingerprint_is_stable_and_sensitive():
    a = Config.load("configs/baseline.toml")
    b = Config.load("configs/baseline.toml")
    assert a.fingerprint() == b.fingerprint()
    c = a.apply_overrides({"generate": {"temperature": 0.7}})
    assert a.fingerprint() != c.fingerprint()


def test_diff_reports_changed_paths():
    a = Config.load("configs/baseline.toml")
    b = a.apply_overrides({"retrieve": {"top_k": 3}})
    diff = a.diff(b)
    paths = {item["path"] for item in diff}
    assert "retrieve.top_k" in paths
    assert len(diff) == 1


def test_experiment_config_load():
    exp = Config.load_experiment("configs/experiments/ab_chunking.toml")
    assert isinstance(exp, ExperimentCfg)
    assert len(exp.variants) >= 2
    assert exp.dataset.endswith(".jsonl")
    assert exp.variants[1].overrides["ingest"]["chunker"] == "fixed"


def test_experiment_requires_two_variants():
    try:
        ExperimentCfg.from_dict({"name": "x", "dataset": "d.jsonl", "variants": [{"name": "only"}]})
    except ConfigError:
        pass
    else:
        raise AssertionError("A/B 实验至少需要两个 variant")
