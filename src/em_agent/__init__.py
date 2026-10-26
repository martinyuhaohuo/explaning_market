"""The live prediction agent: a langgraph graph from an event's materials to a percentile.

With a preview (the competition's, or our own made before the knowledge cutoff):
extract surprises -> classify them into the v2 taxonomy -> build the feature
vector -> score it with the pooled ridge model. Without any preview: one direct
LLM prediction.

Its prompts, schemas and feature code are frozen copies kept here, so changes
to the research code in `suprise_dim` never alter it. The fitted model
(`artifacts/pooled_ridge_v2.joblib`, with the frozen taxonomy) is loaded with
`ml.model`, so `ml` ships with the agent.

- `materials`: reading the event's materials bundle.
- `schemas`, `prompts`: the LLM output schemas and prompts.
- `features`: surprises -> feature vector, with the preview-source flag.
- `context`, `state`, `nodes`, `graph`: the langgraph agent.
- `tracing`: optional Langfuse callbacks.
"""
