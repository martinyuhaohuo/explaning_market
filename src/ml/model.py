"""Ridge regression with time-ordered cross-validation.

Two target modes:
- "raw": fit target_pct directly.
- "residual": fit what target_pct leaves unexplained after an OLS on
  earnings_surprise_pct within the training rows. ΔR² rewards only what a
  prediction adds beyond the ranked earnings surprise, so this aims the model
  at exactly that. The surprise is used for training only; prediction never
  needs it.

Hyperparameters are chosen by the pooled out-of-fold ΔR²: ΔR² ignores a
prediction's scale, so a mean of per-fold scores cannot penalize predictions
shrunk toward a constant and keeps rewarding ever larger penalties.

Predictions are shifted to a percentile scale and clipped to [0, 1], as the
competition requires; ΔR² is unaffected by the shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .metrics import delta_r2, spearman
from .preprocess import build_preprocessor

TARGET = "target_pct"
ES = "earnings_surprise_pct"  # the competition control: within-quarter percentile rank
TIME = "event_datetime"


def make_pipeline(features: list[str], alpha: float, *, log_counts: bool = True) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(features, log_counts=log_counts)), ("ridge", Ridge(alpha=alpha))])


def residualize(train: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """target - OLS(target ~ ES) on the training rows; missing surprises take the median."""
    es = train[ES].fillna(train[ES].median()).to_numpy()
    y = train[TARGET].to_numpy()
    X = np.column_stack([np.ones(len(y)), es])
    (intercept, slope), *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - (intercept + slope * es), intercept, slope


def time_folds(frame: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds in time order: fold k trains on blocks 0..k and validates on block k+1.

    Blocks are cut on release time, so events released at the same moment stay together.
    """
    rank = pd.to_datetime(frame[TIME]).rank(method="min").to_numpy()
    block = np.floor((rank - 1) * (n_splits + 1) / len(frame)).astype(int)
    positions = np.arange(len(frame))
    return [(positions[block <= k], positions[block == k + 1]) for k in range(n_splits)]


@dataclass
class FittedModel:
    pipeline: Pipeline
    features: list[str]
    target_mode: str
    center: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Percentile predictions in [0, 1]."""
        return np.clip(self.pipeline.predict(frame[self.features]) + self.center, 0.0, 1.0)

    def coefficients(self) -> pd.Series:
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        coef = self.pipeline.named_steps["ridge"].coef_
        return pd.Series(coef, index=names).sort_values(key=np.abs, ascending=False)


def fit(train: pd.DataFrame, features: list[str], alpha: float, *, target_mode: str = "raw", log_counts: bool = True) -> FittedModel:
    if target_mode == "residual":
        y, _, _ = residualize(train)
        center = 0.5
    elif target_mode == "raw":
        y = train[TARGET].to_numpy()
        center = 0.0
    else:
        raise ValueError(f"target_mode must be 'raw' or 'residual', not {target_mode!r}")
    pipeline = make_pipeline(features, alpha, log_counts=log_counts).fit(train[features], y)
    return FittedModel(pipeline, features, target_mode, center)


def out_of_fold(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
    *,
    target_mode: str = "raw",
    log_counts: bool = True,
    n_splits: int = 5,
    predict_on: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    """Out-of-fold predictions (column `oof`, NaN for the first block, which is never validated) and per-fold ΔR².

    Returns `train` sorted by time with the `oof` column added, so two runs on
    the same events line up row by row. With `predict_on` (the same events
    under another version of the features, e.g. extracted without a preview),
    each fold is fitted on `train` and predicts the validation events from
    `predict_on`, which measures a model applied to features it was not fitted on.
    """
    train = train.sort_values(TIME).reset_index(drop=True)
    source = train
    if predict_on is not None:
        source = predict_on.set_index("event_id").loc[train["event_id"]].reset_index()
    oof = np.full(len(train), np.nan)
    folds = []
    for fit_idx, val_idx in time_folds(train, n_splits):
        model = fit(train.iloc[fit_idx], features, alpha, target_mode=target_mode, log_counts=log_counts)
        val = source.iloc[val_idx]
        pred = model.predict(val)
        oof[val_idx] = pred
        folds.append(delta_r2(val[TARGET], pred, val[ES])["delta_r2"])
    return train.assign(oof=oof), folds


def cross_validate(
    train: pd.DataFrame,
    features: list[str],
    alpha: float,
    *,
    target_mode: str = "raw",
    log_counts: bool = True,
    n_splits: int = 5,
) -> dict:
    """Per-fold and pooled out-of-fold ΔR² and Spearman for one configuration."""
    train, folds = out_of_fold(train, features, alpha, target_mode=target_mode, log_counts=log_counts, n_splits=n_splits)
    oof = train["oof"].to_numpy()
    scored = ~np.isnan(oof)
    pooled = delta_r2(train[TARGET][scored], oof[scored], train[ES][scored])
    return {
        "cv_delta_r2_mean": float(np.mean(folds)),
        "cv_delta_r2_sd": float(np.std(folds, ddof=1)),
        "oof_delta_r2": pooled["delta_r2"],
        "oof_spearman": spearman(oof[scored], train[TARGET][scored]),
        "fold_delta_r2": folds,
    }


def grid_search(
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    alphas,
    *,
    target_modes=("raw", "residual"),
    log_counts_options=(True, False),
    n_splits: int = 5,
) -> pd.DataFrame:
    """Cross-validate every combination; sorted by pooled out-of-fold ΔR²."""
    rows = []
    for (set_name, features), alpha, mode, log_counts in product(feature_sets.items(), alphas, target_modes, log_counts_options):
        result = cross_validate(train, features, alpha, target_mode=mode, log_counts=log_counts, n_splits=n_splits)
        rows.append({"feature_set": set_name, "n_features": len(features), "alpha": alpha,
                     "target_mode": mode, "log_counts": log_counts, **result})
    return pd.DataFrame(rows).sort_values("oof_delta_r2", ascending=False).reset_index(drop=True)


def rank_to_percentile(score) -> np.ndarray:
    """Turn any score into a [0, 1] percentile by rank (for single-feature reference predictions)."""
    s = pd.Series(np.asarray(score, dtype=float))
    return ((s.rank(method="average") - 1) / (s.notna().sum() - 1)).to_numpy()


def save_model(fitted: FittedModel, path, **metadata) -> None:
    """Write the fitted model (`<path>`, joblib) and a JSON sidecar (`<path>.json`) describing it."""
    import json
    from pathlib import Path

    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, path)
    meta = {"features": fitted.features, "target_mode": fitted.target_mode, "center": fitted.center,
            "sklearn_version": sklearn.__version__, **metadata}
    Path(f"{path}.json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path) -> FittedModel:
    """Load a model written by `save_model`; warns if scikit-learn's version differs from training."""
    import json
    import warnings
    from pathlib import Path

    import joblib
    import sklearn

    meta_path = Path(f"{path}.json")
    if meta_path.exists():
        trained_with = json.loads(meta_path.read_text()).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(f"model trained with scikit-learn {trained_with}, loading with {sklearn.__version__}")
    return joblib.load(path)

