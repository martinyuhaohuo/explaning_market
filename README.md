# Explaining Markets — earnings-surprise agent

An entry for the [Explaining Markets competition](https://explainingmarkets.ai/).
For each earnings release, `predict(event)` returns a percentile in `[0, 1]`: where
the stock's next-day abnormal return will rank among all of the quarter's events.
Entries are scored by **ΔR²**: how much the prediction adds to a regression of
that rank on the ranked EPS surprise (Koijen & Levy). A prediction counts only
for what the headline earnings surprise does not already explain.

The idea: read the earnings call against what was expected before it (the
earnings preview), list every way the results surprised, sort the surprises into
a fixed taxonomy, and score them with a ridge model.

## Workflow

<p align="center">
  <img src="docs/images/workflow_v3_2.png" alt="em_agent workflow, version 3.2. Before the knowledge cutoff, a job checks the competition calendar every 15 minutes and, for every upcoming event, generates our own earnings preview with the Claude Agent SDK about two hours before the cutoff. At release, the event's earnings-call summary and preview (the competition's, else our own) go to a surprise-extraction step; the surprises are classified into a frozen 15-category taxonomy and aggregated into per-event features with an own-preview flag, which a pooled ridge model scores together with option statistics. An event with no preview at all gets one direct gpt-6.1-sol prediction instead. The output, clipped to [0, 1], is the final prediction." width="900">
</p>

Version 3.2 of the workflow (slide 5 of
[`docs/surprise_pipeline_workflow.pptx`](docs/surprise_pipeline_workflow.pptx)),
which matches the code. Solid boxes are implemented; dashed boxes (stock-price
inputs, tuning the prompt with Agentic Context Engineering) are planned.

**Before the cutoff (shadow mode).** A Modal cron job checks the competition
calendar (`GET /v1/events`) every 15 minutes. From 12 hours before each event's
knowledge cutoff, it runs the competition's own earnings-preview skill through the
Claude Agent SDK with a cheap preset (Claude Haiku 5.5, low effort, at most $0.10,
4 searches, 3 page fetches). It does this for **every** event, because whether the
competition will supply a preview is only known at release. At most 5 previews
run at once; the rest wait in Modal's queue. Most cutoffs fall at 16:00 ET, so
each day's previews start around 04:00 ET and are worked through over the
morning. A preview counts only if it finished before the cutoff.

**At release (`predict`).** The event's materials bundle holds the
earnings-call facts, the competition's preview when there is one, and the
option-implied statistics.

1. **Pick the preview:** the competition's if present, else our own.
2. **Extract surprises** (gpt-6-luna, medium effort): every way the call
   differed from expectations, most important first. Each surprise has a
   direction (+1, −1, 0 mixed or unclear), a magnitude (1–3), and quoted
   call and expectation evidence.
3. **Classify surprises** (gpt-6-luna, low effort, one call per event) into the
   frozen v2 taxonomy: 15 categories plus `other`.
4. **Aggregate** into one feature row per event:
   - net direction × magnitude per category, plus its positive and negative parts;
   - totals (counts, net, net of the first three surprises) and a no-surprise flag;
   - option statistics, with missing values imputed;
   - a flag for whether the preview was our own.
5. **Predict** with the pooled ridge model, clipped to `[0, 1]`.

An event with no preview at all happens only when our own preview also failed.
It skips the surprise pipeline and gets one **direct prediction** (gpt-6.1-sol,
medium effort) from the call facts and option statistics.

The **pooled** ridge serves both preview sources. It was trained on every 2026Q3
preview event twice: once with the competition's preview and once with a
condensed "thin" version that imitates ours. The own-preview flag lets it weigh
the two differently.

Every event's previews, surprises, feature vector and prediction are saved to a
Modal Volume. `shadow.sync.download()` copies them to `data/surprise/live/`.

## Repository layout

```
predict.py           the competition entry point: fetch the bundle, run em_agent, record the result
modal_app.py         Modal app: webhook receiver, prediction worker, shadow preview cron job
src/em_agent/        the live agent (what runs in production)
src/shadow/          shadow mode: scheduled own previews and per-event records
src/preview_agent/   our own earnings previews with the Claude Agent SDK
src/suprise_dim/     research: surprise extraction, taxonomy discovery, datasets, features
src/ml/              research: ridge model, time-ordered CV, the ΔR² metric
src/explaining_markets/  competition plumbing (config, webhook verifier, API client)
script/              research and test notebooks
docs/                workflow deck and images
tests/               offline tests (no API keys, no network)
```

### Modules

**`em_agent`: the live prediction agent.** It is a langgraph graph. Its prompts,
schemas and feature code are frozen copies of the research versions, so later
changes to `suprise_dim` cannot change it. Tests check that the copies still
match.

| Module | Purpose |
|---|---|
| `graph.py` | The graph: route by preview → extract → classify → features → ridge, or a direct prediction. |
| `context.py` | `build_context`: attaches the live bundle to the event and picks the preview (competition's, own, or none). |
| `state.py` | The graph state: surprises, classified items, features, prediction, branch. |
| `nodes.py` | The graph steps, the cached LLM clients (gpt-6-luna, gpt-6.1-sol) and the loaders for the model and taxonomy. |
| `prompts.py`, `schemas.py` | Extraction, classification and direct-prediction prompts and their structured-output schemas. |
| `materials.py` | Reads the bundle: call facts, option statistics, ticker. |
| `features.py` | Turns classified surprises into the model's feature row, including the preview-source flag. |
| `tracing.py` | Optional Langfuse tracing; switches off when the host is unreachable. |
| `artifacts/` | The deployed pooled ridge (`pooled_ridge_v2.joblib` plus a description) and the frozen v2 taxonomy. |

**`shadow`: shadow mode on Modal.**

| Module | Purpose |
|---|---|
| `schedule.py` | Finds the calendar events due for a preview (from 12 h before cutoff) and claims each so only one worker runs it. At most 5 previews run at once. |
| `runner.py` | Generates and stores one own preview, recording whether it finished before the cutoff. |
| `store.py` | One folder per event (`events/<event_id>/`) on the Volume or a local directory. |
| `sync.py` | Downloads the Volume and loads predictions, features and preview runs into tables. |

**`preview_agent`: our own earnings previews.**

| Module | Purpose |
|---|---|
| `events.py` | The competition calendar (`GET /v1/events`): fetch, tabulate, pick the next event. |
| `generator.py` | Runs the earnings-preview skill with the Claude Agent SDK. Presets set the model, effort, turns, budget and search/fetch caps, and the run is stopped at the cutoff. |
| `names.py` | Looks up company names for tickers on Yahoo Finance, cached in `data/surprise/company_names.json`. |
| `workspace/` | The agent's working folder with the skill (`.claude/skills/earnings-preview/SKILL.md`). |

**`suprise_dim`: research pipeline for surprise dimensions.**

| Module | Purpose |
|---|---|
| `data.py` | Loads the archive; builds the chronological 2026Q3 train/test split and the discovery sample. |
| `extraction.py` | Batch surprise extraction (with and without a preview), resumable JSONL output. |
| `grouping.py` | Proposes and refines the taxonomy, and assigns surprises to it in resumable batches. |
| `dataset.py` | Names each event set's files and joins them into one row-per-surprise table. |
| `features.py` | Builds the event-level feature table from the surprise table. |
| `labeling.py` | Experimental separate labelling step (direction, magnitude, management sentiment). |
| `thin_preview.py` | Condenses competition previews into thin ones in the style of our own (Claude Haiku 5.5). |
| `prompts.py`, `schemas.py` | Research prompts and output schemas. |

**`ml`: modelling and the metric.**

| Module | Purpose |
|---|---|
| `metrics.py` | The competition's ΔR² with `earnings_surprise_pct` as the control, plus comparison helpers. |
| `preprocess.py` | Feature sets; median imputation with missing-value flags for option statistics; scaling; preview-source features. |
| `model.py` | Ridge pipeline, time-ordered expanding-window CV, grid search by pooled out-of-fold ΔR², save and load. |

**`explaining_markets`: competition plumbing.** Reads the config, verifies
webhook signatures, submits predictions, and has small helpers for event payloads.

### Notebooks (`script/`)

They are listed in the order the research was done. Notebooks that call LLMs
append their results to files and can resume where they stopped.

| Notebook | Purpose |
|---|---|
| `suprise_dim_discovery.ipynb` | Splits 2026Q3 preview events 80/20 by time, samples 500 discovery events, extracts surprises, discovers and refines the taxonomy (v2), and checks how consistently surprises are assigned. |
| `ridge_model.ipynb` | Builds event features, fits ridge with time-ordered CV, and compares ΔR² on test against the competition baselines. |
| `label_split.ipynb` | Tests a separate labelling step with management sentiment against the extraction's own labels. It was not worth adding. |
| `no_preview.ipynb` | Extraction without a preview: how much ΔR² is lost when the preview is hidden from train events, and how best to score the 997 Q3 events that never had one. |
| `earnings_preview.ipynb` | Generates our own preview for the next scheduled event with the Claude Agent SDK. |
| `thin_preview.ipynb` | Condenses competition previews into thin ones, re-extracts surprises from them, and compares separate and pooled models. |
| `train_pooled_model.ipynb` | Trains the deployed pooled ridge on all events and writes it and the frozen taxonomy to `src/em_agent/artifacts/`. |
| `basic_agent_test.ipynb` | Paid end-to-end dry run of `predict()` on two archived 2026Q4 events (one per branch), with records written to `data/surprise/dry_run/`. |

## Deploy

### Prerequisites

This repo uses [uv](https://docs.astral.sh/uv/) — install it from the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

> **Prefer pip?** Run `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
> instead of `uv sync`, and drop the `uv run` prefix from every command below.

### 0. Install and sign in to Modal

```bash
uv sync
```

If you're new to Modal, create a free account and authenticate (one time — skip if
you already have a Modal token on this machine):

```bash
uv run modal setup
```

### 1. Create an account and your first submission

Go to [Explaining Markets](https://portal.explainingmarkets.ai) and click
**Sign in** at the top right, then **Create an account**, and complete the
sign-up flow.

Once you're in, create a submission from the
[Submissions](https://portal.explainingmarkets.ai/submissions) page and
give it a public name. You'll land on its **Overview** tab, which has a
**Setup checklist** that walks you through the rest:

> **Credentials → Webhook URL → Submission is live → Verify your endpoint works**

The next steps map onto that checklist; the submission goes live automatically once
the first two are done.

### 2. Initialize credentials (checklist: *Credentials*)

Click **Initialize credentials** (the checklist's first item) to mint your **API
key** and **webhook signing secret**. A dialog shows them **once**, under the
heading *"Ready to paste into .env"*, already formatted — exactly the two lines this
app needs:

> `EM_API_KEY=...`
>
> `EM_WEBHOOK_SECRET=whsec_...`

Click **Copy** (the dialog won't let you continue until you do) — you won't see
these again, so don't close it before the next step. (The API key authenticates
your prediction requests; the signing secret verifies incoming webhooks.)

**Note:** If you ever need new credentials — e.g., because they were accidentally
leaked — that same item becomes **Replace credentials**. Clicking it mints a new set
you can use to continue from **Step 3**.

### 3. Put your credentials in `.env`

Create your `.env` from the template, then paste the copied box into it, replacing
the two placeholder lines:

```bash
cp .env.example .env
```

Modal reads `.env` automatically at deploy time, so there's no command to run.
(`.env` is gitignored; never commit it.) Add these provider keys as well:

- `OPENAI_API_KEY`: the agent's extraction, classification and direct
  prediction. Without it `predict()` submits the neutral 0.5.
- `ANTHROPIC_API_KEY`: shadow mode's own previews (Claude Agent SDK).
- `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: optional
  tracing; off when blank or when the host is unreachable.

### 4. Deploy

```bash
uv run modal deploy modal_app.py
```

Modal prints a persistent public URL like
`https://<your-workspace>--explaining-markets.modal.run`. **That URL is your webhook
URL — copy it as-is, nothing to append.** The deployment keeps running after you
close your laptop. It also starts the shadow job (`schedule_previews`, every 15
minutes) and creates the `em-shadow-data` Volume for the records.

### 5. Set your webhook URL, go live, and verify

Back on the Overview checklist, do **Webhook URL**: paste the URL from the previous
step (it must be reachable over HTTPS in production; `http://` is allowed in dev)
and click **Save webhook URL**. As soon as credentials and a URL are both set,
**Submission is live** flips on automatically — no extra action.

The last item, **Verify your endpoint works**, is optional but strongly
recommended. Click **Send test event** to send a synthetic delivery. Your
handler verifies it, sees `event_type == "TEST"`, submits a neutral 0.5
prediction back (test predictions are never scored), and ACKs with 200; the
checklist confirms your endpoint responded and your prediction came back. If
nothing appears right away, check the **Health** tab for rolling delivery
counters.

Re-deploy after any change to the code or `.env`:

```bash
uv run modal deploy modal_app.py
```
