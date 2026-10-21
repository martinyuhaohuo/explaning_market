import os
import subprocess
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END

from .state import AnalysisState
from .context import AnalysisContext
from .prompts import ANALYSIS_SYSTEM, CRITIQUE_SYSTEM
from .schemas import Analysis, Critique

load_dotenv()
try:
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except:
    print("LLM API key not found")

FORMAT_MODEL = "gemini-3.1-flash-lite-preview"
PROPOSE_MODEL  = "gpt-6-luna"
CRITIQUE_MODEL = "gpt-6.1-sol"

proposer = ChatOpenAI(
    model = PROPOSE_MODEL,
    reasoning = {"effort": "low", "summary": "auto"},
    api_key = OPENAI_API_KEY,
)

critic = ChatOpenAI(
    model = CRITIQUE_MODEL,
    reasoning = {"effort": "medium", "summary": "auto"},
    api_key = OPENAI_API_KEY,
)

formatter_llm = ChatGoogleGenerativeAI(
    model=FORMAT_MODEL,
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
)

def make_formatter(schema, input_field: str, output_field: str, tag: str):
    """Build a node that converts state[input_field] (str) into a `schema` instance
    and writes it to state[output_field]. Tags the LLM call so the logger
    can attribute it."""
    structured = formatter_llm.with_structured_output(schema)
    sys = SystemMessage(
        f"You are a formatter. Convert the user's text into a {schema.__name__} "
        f"object. Preserve all substantive content. Do not invent new facts. "
        f"If a field is not explicitly stated, infer it conservatively from context."
    )

    def formatter(state, runtime):
        raw = state[input_field]
        obj = structured.invoke([sys, HumanMessage(raw)], config={"tags": [tag]})
        return {output_field: obj}
    return formatter

def analysis_maker(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> AnalysisState:
    ctx = runtime.context
    round_n = state.get("round", 0) + 1
    critique = state.get("current_critique")
    if ctx.earnings_preview is not None:
        earnings_preview = str(ctx.earnings_preview)
    else:
        earnings_preview = "Not avaliable"
    if ctx.option_stats is not None:
        option_stats = str(ctx.option_stats)
    else:
        option_stats = "Not avaliable"

    if critique is None:
        user = (
            f"Firm stock ticker:\n{ctx.stock_id}\n\n"
            f"Earning call facts:\n{ctx.earnings_call_facts}\n\n"
            f"Earning preview:\n{earnings_preview}\n\n"
            f"Option relevant statistics:\n{option_stats}\n\n"
            f"Constraint:\n{ctx.constraints}\n\n"
            f"Analyze the given information and provide your prediction."
        )
    else:
        prev = state["current_analysis"]
        user = (
            f"Firm stock ticker:\n{ctx.stock_id}\n\n"
            f"Earning call facts:\n{ctx.earnings_call_facts}\n\n"
            f"Earning preview:\n{earnings_preview}\n\n"
            f"Option relevant statistics:\n{option_stats}\n\n"
            f"Constraint:\n{ctx.constraints}\n\n"
            f"Your previous analysis:\n{prev.model_dump_json(indent=2)}\n\n"
            f"Vice president's critique:\n{critique.model_dump_json(indent=2)}\n\n"
            f"Revise. Return one revised analysis toegther with your prediction."
        )

    msg = proposer.invoke(
        [SystemMessage(ANALYSIS_SYSTEM), HumanMessage(user)],
        config={"tags": ["analysis_maker"]},
    )
    return {"raw_analysis": msg.text, "round": round_n}

def analysis_evaluator(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> AnalysisState:
    ctx = runtime.context
    analysis = state["current_analysis"]
    if ctx.earnings_preview is not None:
        earnings_preview = str(ctx.earnings_preview)
    else:
        earnings_preview = "Not avaliable"
    if ctx.option_stats is not None:
        option_stats = str(ctx.option_stats)
    else:
        option_stats = "Not avaliable"
    user = (
        f"Firm stock ticker:\n{ctx.stock_id}\n\n"
        f"Earning call facts:\n{ctx.earnings_call_facts}\n\n"
        f"Earning preview:\n{earnings_preview}\n\n"
        f"Option relevant statistics:\n{option_stats}\n\n"
        f"Constraint:\n{ctx.constraints}\n\n"
        f"Analysis and prediction under review:\n{analysis.model_dump_json(indent=2)}\n\n"
        f"Evaluate it."
    )
    msg = critic.invoke(
        [SystemMessage(CRITIQUE_SYSTEM), HumanMessage(user)],
        config={"tags": ["analysis_evaluator"]},
    )
    return {"raw_critique": msg.text}

format_analysis = make_formatter(
    schema=Analysis,
    input_field="raw_analysis",
    output_field="current_analysis",
    tag="format_idea",
)


format_critique = make_formatter(
    schema=Critique,
    input_field="raw_critique",
    output_field="current_critique",
    tag="format_critique",
)

def route_after_format_critique(state: AnalysisState, runtime: Runtime[AnalysisContext]) -> str:
    if state["round"] < runtime.context.num_rounds:
        return "analysis_maker"
    return END