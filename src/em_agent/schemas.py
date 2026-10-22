from pydantic import BaseModel, Field

class Analysis(BaseModel):
    """A concrete analysis."""
    prediction: float = Field(
        ge=0.0,
        le=1.0,
        description="A single percentile in [0, 1].",
    )
    reasons: str = Field(description="3-5 reasons that support your prediction.")

class Critique(BaseModel):
    """Structured critique of an analysis."""
    agree_prediction: bool = Field(description="True if the analysis sounds reaonsable.")
    limitation: str = Field(description="What are the limitations of the analysis, under current constraint.")
    suggestion: list[str] = Field(description="Concrete suggestions to make on the next revision.")

def _type_name(ann) -> str:
    # str(int) -> "<class 'int'>", str(list[str]) -> "list[str]" — clean both.
    s = str(ann).replace("typing.", "")
    if s.startswith("<class '") and s.endswith("'>"):
        return s[len("<class '"):-len("'>")]
    return s


def schema_field_brief(schema) -> str:
    """Render a Pydantic schema as a bullet list of fields + descriptions.

    Used inside *generator* prompts so the writer knows what to cover, while
    still producing free-form prose (no JSON)."""
    lines = []
    for name, field in schema.model_fields.items():
        t = _type_name(field.annotation)
        desc = field.description or ""
        lines.append(f"- {name} ({t}): {desc}")
    return "\n".join(lines)
