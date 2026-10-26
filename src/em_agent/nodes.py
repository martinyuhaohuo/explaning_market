"""Graph nodes. LLM clients and artifacts are built once and cached; tests patch the factories."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from ml.model import FittedModel, load_model

from .context import SurpriseContext
from .features import live_features
from .prompts import (
    ASSIGN_SYSTEM, DIRECT_SYSTEM, EXTRACTION_SYSTEM, build_direct_message, build_extraction_message,
    format_items, format_taxonomy,
)
from .schemas import OTHER, Analysis, SurpriseExtraction, Taxonomy, assignment_schema
from .state import SurpriseState

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS / "pooled_ridge_v2.joblib"
TAXONOMY_PATH = ARTIFACTS / "taxonomy_v2.json"

EXTRACT_MODEL, EXTRACT_EFFORT = "gpt-6-luna", "medium"
ASSIGN_MODEL, ASSIGN_EFFORT = "gpt-6-luna", "low"
DIRECT_MODEL, DIRECT_EFFORT = "gpt-6.1-sol", "medium"


@lru_cache
def taxonomy() -> Taxonomy:
    return Taxonomy.model_validate_json(TAXONOMY_PATH.read_text())


@lru_cache
def pooled_model() -> FittedModel:
    return load_model(MODEL_PATH)


@lru_cache
def extractor():
    return ChatOpenAI(model=EXTRACT_MODEL, reasoning={"effort": EXTRACT_EFFORT}).with_structured_output(SurpriseExtraction)


@lru_cache
def assigner():
    schema = assignment_schema([d.key for d in taxonomy().dimensions])
    return ChatOpenAI(model=ASSIGN_MODEL, reasoning={"effort": ASSIGN_EFFORT}).with_structured_output(schema)


@lru_cache
def direct_predictor():
    return ChatOpenAI(model=DIRECT_MODEL, reasoning={"effort": DIRECT_EFFORT}).with_structured_output(Analysis)


def route_by_preview(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> str:
    return "extract_surprises" if runtime.context.preview_source else "direct_predict"


def extract_surprises(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> dict:
    event = runtime.context.event
    result = extractor().invoke(
        [SystemMessage(EXTRACTION_SYSTEM), HumanMessage(build_extraction_message(event))],
        config={"tags": ["surprise_extraction"], "metadata": {"event_id": event["event_id"]}},
    )
    return {"surprises": [s.model_dump() for s in result.surprises]}


def classify_surprises(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> dict:
    """One call for the event's surprises; any the model skips or misnumbers count as "other"."""
    surprises = state.get("surprises") or []
    items = [{"position": i, **s, "dimension": OTHER} for i, s in enumerate(surprises)]
    if items:
        user = f"Codebook:\n{format_taxonomy(taxonomy())}\n\nItems:\n{format_items([s['description'] for s in surprises])}"
        result = assigner().invoke([SystemMessage(ASSIGN_SYSTEM), HumanMessage(user)], config={"tags": ["assign_dimensions"]})
        for a in result.assignments:
            if 0 <= a.index < len(items):
                items[a.index]["dimension"] = a.dimension
    return {"items": items}


def build_features(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> dict:
    ctx = runtime.context
    return {"features": live_features(ctx.event, state.get("items") or [], taxonomy(), ctx.preview_source)}


def ridge_predict(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> dict:
    prediction = pooled_model().predict(pd.DataFrame([state["features"]]))[0]
    return {"prediction": float(prediction), "branch": "surprise_ridge"}


def direct_predict(state: SurpriseState, runtime: Runtime[SurpriseContext]) -> dict:
    event = runtime.context.event
    result = direct_predictor().invoke(
        [SystemMessage(DIRECT_SYSTEM), HumanMessage(build_direct_message(event))],
        config={"tags": ["direct_prediction"], "metadata": {"event_id": event["event_id"]}},
    )
    return {"prediction": result.prediction, "reasons": result.reasons, "branch": "direct"}
