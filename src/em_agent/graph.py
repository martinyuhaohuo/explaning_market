from langgraph.graph import StateGraph, START, END
from .state import AnalysisState
from .context import AnalysisContext
from .nodes import analysis_maker, format_analysis, analysis_evaluator, format_critique, route_after_format_critique


graph = (
    StateGraph(AnalysisState, context_schema=AnalysisContext)
    .add_node("analysis_maker", analysis_maker)
    .add_node("format_analysis", format_analysis)
    .add_node("analysis_evaluator", analysis_evaluator)
    .add_node("format_critique", format_critique)
    .add_edge(START, "analysis_maker")
    .add_edge("analysis_maker", "format_analysis")
    .add_edge("format_analysis", "analysis_evaluator")
    .add_edge("analysis_evaluator", "format_critique")
    .add_conditional_edges(
        "format_critique",
        route_after_format_critique,
        {"analysis_maker": "analysis_maker", END: END},
    )
    .compile()
)