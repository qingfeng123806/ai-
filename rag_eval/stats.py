"""统计工具：均值、分位数、Bootstrap 置信区间、配对 Bootstrap 显著性检验。

纯标准库实现，保证在无 scipy/numpy 的环境下也能给出可信的区间估计。
A/B 对比使用配对 Bootstrap（同一批测试用例上两组结果的差值重采样），
比独立样本 t 检验更适合「同一测试集、同一问题」的实验设计。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def bootstrap_ci(
    values: Sequence[float],
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int | str = 0,
) -> tuple[float, float]:
    """对均值做 Bootstrap 百分位区间。样本过少或未开启时退化为点估计。"""
    if not values:
        return 0.0, 0.0
    if len(values) < 3 or samples <= 0:
        m = mean(values)
        return m, m
    rng = random.Random(str(seed))
    n = len(values)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    alpha = (1.0 - confidence) / 2.0
    return percentile(means, alpha), percentile(means, 1.0 - alpha)


def paired_bootstrap_pvalue(
    left: Sequence[float],
    right: Sequence[float],
    samples: int = 2000,
    seed: int | str = 0,
) -> float:
    """配对 Bootstrap 双侧 p 值：H0 为两组均值差为 0。"""
    n = min(len(left), len(right))
    if n == 0 or samples <= 0:
        return 1.0
    diffs = [right[i] - left[i] for i in range(n)]
    observed = mean(diffs)
    if all(abs(d) < 1e-12 for d in diffs):
        return 1.0
    # 关键：零假设下差值应以 0 为中心，因此先中心化再重采样，
    # 直接对原始差值重采样会让 p 值恒在 0.5 附近（经典实现错误）。
    centered = [d - observed for d in diffs]
    rng = random.Random(f"paired:{seed}")
    extreme = 0
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += centered[rng.randrange(n)]
        boot = total / n
        if abs(boot) >= abs(observed):
            extreme += 1
    return min(1.0, max(1.0 / (samples + 1), extreme / samples))


def cohens_d(left: Sequence[float], right: Sequence[float]) -> float:
    """效应量：两组均值差除以合并标准差。"""
    n1, n2 = len(left), len(right)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = stdev(left), stdev(right)
    pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / max(1, n1 + n2 - 2))
    if pooled <= 1e-12:
        return 0.0
    return (mean(right) - mean(left)) / pooled


@dataclass
class DeltaResult:
    metric: str
    left: float
    right: float
    delta: float
    relative: float
    p_value: float
    significant: bool
    effect_size: float
    better: str          # "left" | "right" | "tie"
    direction: str = "higher"
    n: int = 0

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "left": round(self.left, 4),
            "right": round(self.right, 4),
            "delta": round(self.delta, 4),
            "relative": round(self.relative, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "effect_size": round(self.effect_size, 3),
            "better": self.better,
            "direction": self.direction,
            "n": self.n,
        }


def compare_series(
    metric: str,
    left: Sequence[float],
    right: Sequence[float],
    direction: str = "higher",
    alpha: float = 0.05,
    samples: int = 2000,
    seed: int | str = 0,
) -> DeltaResult:
    l_mean = mean(left)
    r_mean = mean(right)
    delta = r_mean - l_mean
    relative = (delta / l_mean) if abs(l_mean) > 1e-12 else 0.0
    p = paired_bootstrap_pvalue(left, right, samples=samples, seed=seed)
    if abs(delta) < 1e-9:
        better = "tie"
    elif direction == "lower":
        better = "right" if delta < 0 else "left"
    else:
        better = "right" if delta > 0 else "left"
    return DeltaResult(
        metric=metric,
        left=l_mean,
        right=r_mean,
        delta=delta,
        relative=relative,
        p_value=p,
        significant=p < alpha,
        effect_size=cohens_d(left, right),
        better=better,
        direction=direction,
        n=min(len(left), len(right)),
    )
