"""With a preview: extract -> classify -> features -> pooled ridge. Without: one direct prediction."""

from langgraph.graph import END, START, StateGraph

from .context import SurpriseContext
from .nodes import build_features, classify_surprises, direct_predict, extract_surprises, ridge_predict, route_by_preview
from .state import SurpriseState

builder = StateGraph(SurpriseState, context_schema=SurpriseContext)
builder.add_node("extract_surprises", extract_surprises)
builder.add_node("classify_surprises", classify_surprises)
builder.add_node("build_features", build_features)
builder.add_node("ridge_predict", ridge_predict)
builder.add_node("direct_predict", direct_predict)

builder.add_conditional_edges(START, route_by_preview, ["extract_surprises", "direct_predict"])
builder.add_edge("extract_surprises", "classify_surprises")
builder.add_edge("classify_surprises", "build_features")
builder.add_edge("build_features", "ridge_predict")
builder.add_edge("ridge_predict", END)
builder.add_edge("direct_predict", END)

graph = builder.compile()
