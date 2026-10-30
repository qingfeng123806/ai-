"""统计工具测试：Bootstrap 区间与配对显著性。"""

from rag_eval.stats import (
    bootstrap_ci,
    cohens_d,
    compare_series,
    mean,
    paired_bootstrap_pvalue,
    percentile,
    stdev,
)


def test_mean_and_percentile():
    assert mean([1, 2, 3]) == 2
    assert mean([]) == 0
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.9) == 5


def test_stdev():
    assert stdev([2, 2, 2]) == 0
    assert abs(stdev([1, 2, 3]) - 1.0) < 1e-9


def test_bootstrap_ci_contains_mean():
    values = [0.4, 0.5, 0.6, 0.7, 0.5, 0.55, 0.45]
    low, high = bootstrap_ci(values, samples=400, confidence=0.9, seed="test")
    assert low <= mean(values) <= high
    assert high - low < 0.6


def test_bootstrap_ci_degenerate_cases():
    assert bootstrap_ci([]) == (0.0, 0.0)
    low, high = bootstrap_ci([0.5, 0.5])
    assert low == high == 0.5


def test_paired_pvalue_extremes():
    left = [0.9, 0.8, 0.7, 1.0, 0.95, 0.85]
    right = [0.4, 0.5, 0.2, 0.6, 0.3, 0.4]
    assert paired_bootstrap_pvalue(left, right, samples=1500) < 0.01
    assert paired_bootstrap_pvalue(left, list(left), samples=500) == 1.0


def test_paired_pvalue_null_is_not_significant():
    left = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.52, 0.48]
    right = [0.51, 0.59, 0.41, 0.54, 0.46, 0.49, 0.53, 0.47]
    assert paired_bootstrap_pvalue(left, right, samples=800) > 0.05


def test_compare_series_direction_aware():
    left = [0.9, 0.8, 0.85, 0.95]
    right = [0.4, 0.5, 0.45, 0.35]
    better = compare_series("faithfulness", left, right, direction="higher", samples=800)
    assert better.better == "left"
    assert better.delta < 0
    assert better.significant

    lower_is_better = compare_series("latency", [100, 110, 105], [50, 55, 45], direction="lower", samples=800)
    assert lower_is_better.better == "right"


def test_cohens_d_sign_and_magnitude():
    assert cohens_d([0.9, 0.8, 0.85], [0.2, 0.3, 0.25]) < -2
    assert cohens_d([0.5, 0.5], [0.5, 0.5]) == 0.0
