from typing import TypedDict, Optional
from .schemas import Critique, Analysis

class AnalysisState(TypedDict, total=False):
    raw_analysis: str
    current_analysis: Analysis
    raw_critique: str
    current_critique: Optional[Critique]
    round: int