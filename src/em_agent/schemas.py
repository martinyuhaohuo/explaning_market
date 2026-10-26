from typing import Literal

from pydantic import BaseModel, Field, create_model


class SurpriseItem(BaseModel):
    """One concrete way the results differ from what was expected."""

    description: str = Field(
        description=(
            "One or two sentences: what was expected, what was reported, and how they "
            "differ. Be specific to this company; do not name an abstract category."
        )
    )
    direction: Literal["positive", "negative", "mixed", "unclear"] = Field(
        description="Whether this surprise is good or bad news for shareholders."
    )
    magnitude: Literal["small", "moderate", "large"] = Field(
        description="Size of the gap relative to the expectation, not its importance."
    )
    fact_evidence: str = Field(description="The earnings-call fact(s) this rests on, quoted closely.")
    expectation_evidence: str = Field(
        description="The preview line stating the expectation, quoted closely. Empty if none."
    )
    expectation_source: Literal["preview", "call", "none"] = Field(
        description=(
            "'preview' if the expectation comes from the preview, 'call' if only the call "
            "itself states it (e.g. 'beat consensus'), 'none' if the expectation is implicit."
        )
    )


class SurpriseExtraction(BaseModel):
    """All surprises in one earnings release."""

    surprises: list[SurpriseItem] = Field(description="Every distinct surprise, most important first.")
    in_line: list[str] = Field(
        description="Expectations the preview flagged as important that came in as expected."
    )


class Dimension(BaseModel):
    """One surprise dimension in the codebook."""

    key: str = Field(description="Short snake_case identifier, unique within the taxonomy.")
    name: str = Field(description="Human-readable name.")
    definition: str = Field(description="One or two sentences: what kind of surprise belongs here.")
    includes: list[str] = Field(description="Two to four short, concrete examples that belong here.")
    excludes: str = Field(
        description="Nearby surprises that do NOT belong here, and which dimension they go to instead."
    )


class Taxonomy(BaseModel):
    """A codebook of surprise dimensions."""

    dimensions: list[Dimension] = Field(description="The dimensions, most frequently used first.")
    changes: list[str] = Field(
        description="What changed relative to the starting list and why, one line per change."
    )


class Analysis(BaseModel):
    """A direct prediction, for events without any preview."""

    prediction: float = Field(ge=0.0, le=1.0, description="A single percentile in [0, 1].")
    reasons: str = Field(description="3-5 reasons that support your prediction.")


OTHER = "other"


def assignment_schema(keys: list[str]):
    """A batch schema whose `dimension` can only be one of the taxonomy's keys or 'other'."""
    key_type = Literal[tuple(keys + [OTHER])]
    item = create_model(
        "ItemAssignment",
        index=(int, Field(description="The item's number.")),
        dimension=(key_type, Field(description="Key of the dimension this item belongs to.")),
    )
    return create_model(
        "AssignmentBatch",
        assignments=(list[item], Field(description="One assignment per item.")),
    )
