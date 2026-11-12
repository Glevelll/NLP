"""Статистическая значимость: bootstrap-доверительные интервалы, парный bootstrap и парный t-тест."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
                 ) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: list[float], b: list[float], n_resamples: int = 2000, seed: int = 0) -> float:
    """p-value для гипотезы «система A не лучше B» (доля выборок, где mean(A-B) <= 0)."""
    da = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if da.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, da.size, size=(n_resamples, da.size))
    return float(np.mean(da[idx].mean(axis=1) <= 0))


def paired_ttest(a: list[float], b: list[float]) -> dict[str, float]:
    from scipy import stats

    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.size < 2 or np.allclose(a_arr - b_arr, 0):
        return {"t": 0.0, "p_value": 1.0}
    res = stats.ttest_rel(a_arr, b_arr)
    return {"t": float(res.statistic), "p_value": float(res.pvalue)}


def compare_systems(per_sample: dict[str, dict[str, list[float]]], metric: str, baseline: str) -> dict[str, dict]:
    """Сравнивает каждую систему с базовой по выбранной метрике."""
    base = per_sample[baseline][metric]
    out = {}
    for name, metrics in per_sample.items():
        if name == baseline or metric not in metrics:
            continue
        vals = metrics[metric]
        n = min(len(vals), len(base))
        out[name] = {
            "delta": float(np.mean(vals[:n]) - np.mean(base[:n])),
            "bootstrap_p": paired_bootstrap(vals[:n], base[:n]),
            **{f"ttest_{k}": v for k, v in paired_ttest(vals[:n], base[:n]).items()},
        }
    return out
