"""Ridge model on event-level surprise features, scored the way the competition scores.

`preprocess` picks feature sets and builds the imputation/transformation step,
`model` fits ridge regression with time-ordered cross-validation, and
`metrics` computes the competition's ΔR² (what a prediction adds to a
regression of the target on the ranked earnings surprise,
`earnings_surprise_pct`) plus a paired bootstrap for comparing predictions.
"""
