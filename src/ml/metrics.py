"""Competition metric and comparison helpers.

ΔR² = R²(target ~ prediction + surprise) - R²(target ~ surprise), both
ordinary least squares with an intercept, where the surprise is the
within-quarter percentile rank of earnings_surprise (`earnings_surprise_pct`),
as in Koijen and Levy. Rows missing any of the three values are dropped, and
the number used is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """R² of an OLS fit of y on the columns of X plus an intercept."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    total = y - y.mean()
    return 1.0 - (residual @ residual) / (total @ total)


def _complete(*arrays) -> list[np.ndarray]:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    keep = np.all([~np.isnan(a) for a in arrays], axis=0)
    return [a[keep] for a in arrays]


def delta_r2(target, prediction, earnings_surprise) -> dict:
    """The competition's ΔR² and its two components."""
    y, pred, es = _complete(target, prediction, earnings_surprise)
    r2_base = ols_r2(y, es[:, None])
    r2_full = ols_r2(y, np.column_stack([pred, es]))
    return {"n": len(y), "r2_base": r2_base, "r2_full": r2_full, "delta_r2": r2_full - r2_base}


def spearman(a, b) -> float:
    a, b = _complete(a, b)
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def evaluate(target, predictions: dict[str, np.ndarray], earnings_surprise) -> pd.DataFrame:
    """ΔR², its components and Spearman with the target for several predictions on the same rows."""
    rows = {}
    for name, pred in predictions.items():
        rows[name] = {**delta_r2(target, pred, earnings_surprise), "spearman": spearman(pred, target)}
    return pd.DataFrame(rows).T.sort_values("delta_r2", ascending=False)


def paired_bootstrap(
    target, predictions: dict[str, np.ndarray], earnings_surprise, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """ΔR² of every prediction on the same bootstrap resamples (rows complete for all of them).

    Returns one row per resample and one column per prediction, so differences
    between columns are paired comparisons.
    """
    names = list(predictions)
    y, es, *preds = _complete(target, earnings_surprise, *predictions.values())
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(names)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        base = ols_r2(y[idx], es[idx, None])
        for j, pred in enumerate(preds):
            out[b, j] = ols_r2(y[idx], np.column_stack([pred[idx], es[idx]])) - base
    return pd.DataFrame(out, columns=names)


def compare_to(samples: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired bootstrap summary of `reference` minus every other prediction."""
    rows = {}
    for name in samples.columns:
        if name == reference:
            continue
        diff = samples[reference] - samples[name]
        rows[name] = {
            "mean_diff": diff.mean(),
            "ci_low": diff.quantile(0.025),
            "ci_high": diff.quantile(0.975),
            f"P({reference} better)": (diff > 0).mean(),
        }
    return pd.DataFrame(rows).T
