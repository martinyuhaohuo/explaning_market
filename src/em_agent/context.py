from dataclasses import dataclass


@dataclass
class AnalysisContext:
    stock_id: str
    earnings_call_date: str
    earnings_call_facts: list[str] | str
    earnings_preview: str | None
    option_stats: dict | str | None
    knowledge_cutoff: str
    num_rounds: int = 2
    constraints: str = "No external data sources, base your judgment on provided information."
