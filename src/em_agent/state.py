from typing import TypedDict


class SurpriseState(TypedDict, total=False):
    surprises: list[dict]  # extracted surprises, most important first
    items: list[dict]  # the same surprises with position and dimension
    features: dict  # the feature vector the model scored
    prediction: float
    reasons: str  # direct branch only
    branch: str  # "surprise_ridge" or "direct"
