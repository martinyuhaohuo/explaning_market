"""Feature sets, missing-value imputation and transformation.

Surprise features are complete (an event without surprises in a dimension
scores 0), so only the option-implied statistics have missing values: they are
imputed with the training median plus a missing-value indicator. Count-like
surprise features can be compressed with a signed log, and everything is then
standardized so the ridge penalty treats features alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

OPTION_COLUMNS = [
    "opt_implied_earnings_volatility",
    "opt_implied_absolute_earnings_move",
    "opt_skew_25_delta",
]
FLAG_COLUMNS = ["no_surprises", "has_option_stats"]
TOTAL_COLUMNS = ["n_items", "n_pos", "n_neg", "n_mixed_unclear", "net", "net_w", "net_first3", "net_w_first3", "net_per_item"]


def signed_log1p(x):
    """sign(x) * log(1 + |x|): keeps sign and zero, compresses large counts."""
    return np.sign(x) * np.log1p(np.abs(x))


def feature_sets(columns) -> dict[str, list[str]]:
    """Named feature lists; none of them include the target, car1, earnings_surprise or baselines.

    - totals: overall scores only (5) + flags and options.
    - net: magnitude-weighted net per dimension (15) + first-3 net, item count + flags and options.
    - posneg: positive and negative parts per dimension (30) + the same extras.
    - all: every surprise feature, flag and option statistic.
    """
    columns = list(columns)
    net_dims = [c for c in columns if c.startswith("net_w__")]
    posneg_dims = [c for c in columns if c.startswith(("pos_w__", "neg_w__"))]
    extras = FLAG_COLUMNS + OPTION_COLUMNS
    return {
        "totals": ["net_w", "net_w_first3", "net_per_item", "n_items", "n_mixed_unclear"] + extras,
        "net": net_dims + ["net_w_first3", "n_items"] + extras,
        "posneg": posneg_dims + ["net_w_first3", "n_items"] + extras,
        "all": net_dims + posneg_dims + TOTAL_COLUMNS + extras,
    }


def build_preprocessor(features: list[str], *, log_counts: bool = True) -> ColumnTransformer:
    """Median imputation + indicator for option statistics; optional signed log for the rest; then scaling."""
    options = [c for c in features if c in OPTION_COLUMNS]
    flags = [c for c in features if c in FLAG_COLUMNS]
    counts = [c for c in features if c not in options and c not in flags]

    count_steps = [("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))] if log_counts else []
    transformers = [
        ("counts", Pipeline(count_steps + [("scale", StandardScaler())]), counts),
        ("flags", StandardScaler(), flags),
    ]
    if options:
        transformers.append((
            "options",
            Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]),
            options,
        ))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def missing_summary(frame: pd.DataFrame, columns: list[str], by: str = "split") -> pd.DataFrame:
    """Share of missing values per column, overall and per group."""
    table = frame.groupby(by)[columns].apply(lambda g: g.isna().mean()).T
    table["all"] = frame[columns].isna().mean()
    return table[table["all"] > 0].round(3)
