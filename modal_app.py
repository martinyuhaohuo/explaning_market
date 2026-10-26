"""Modal deployment for the Explaining Markets starter.

This is plumbing — you shouldn't need to edit it. It defines a small FastAPI app
and deploys it as a persistent, public web endpoint:

    GET  /    health check
    POST /    receive a signed event, verify, ACK, then predict and submit
              (POST /competition/webhook is kept as an alias of the same handler)

The webhook is served at the root path on purpose: the URL Modal prints on deploy
*is* your webhook URL — paste it into the portal as-is, nothing to append.

Deploy:    uv run modal deploy modal_app.py
Dev/local: uv run modal serve modal_app.py

The webhook handler ACKs first, then predicts. It verifies the signature, returns
200, and spawns `predict_and_submit` — a separate Modal function with its own
container — to run your `predict()` from predict.py and POST the result. Two
clocks:

  * 20 seconds to ACK the delivery. Miss it and the platform retries; repeated
    failures disable your webhook.
  * 5 minutes from that ACK to submit your prediction.

Predicting before the ACK spends the 5-minute budget inside the 20-second one.
Spawning rather than using a background task also means the work doesn't depend
on the web container staying alive.

Deliveries are deduped on the `Webhook-Id` header (the server retries on
4xx/5xx/timeout, so the same event can arrive more than once).

Shadow mode (see src/shadow): `schedule_previews` runs every 15 minutes, finds
calendar events whose knowledge cutoff is less than 12 hours away, and spawns
`generate_own_preview` for each (at most 5 run at once; the rest queue), which runs the earnings-preview skill with the
Claude Agent SDK (cheap preset, stopped at the cutoff). At release, predict()
uses the competition's preview when the event has one, otherwise ours. Every
event's previews, surprises, feature vector and prediction are written to the
`em-shadow-data` Volume; download with `shadow.sync.download()` or
`uv run modal volume get em-shadow-data / data/surprise/live`.

Note: we deliberately do NOT use `from __future__ import annotations` here. The
route handlers are defined inside `web()`, and FastAPI must see the real `Request`
/ `Response` classes (not stringized annotations it can't resolve from this nested
scope) to inject them correctly — otherwise it treats `request` as a query
parameter and rejects every delivery with 422.
"""

import modal

from shadow.schedule import MAX_PARALLEL_PREVIEWS

app = modal.App("explaining-markets-starter")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "httpx",
        "openai",
        "pydantic",
        "python-dotenv",
        "langgraph>=1.2.14",
        "langchain>=1.4.4",
        "langchain-openai>=1.7.0",
        "langchain-google-genai>=4.4.1",
        "langfuse>=4.17.0",
        # Pooled ridge model. scikit-learn is pinned to the version that
        # trained src/em_agent/artifacts/pooled_ridge_v2.joblib.
        "scikit-learn==1.9.1",
        "joblib==1.6.0",
        "pandas==3.0.6",
        "numpy==2.5.4",
        # Shadow previews: the Agent SDK ships its own Claude Code binary.
        "claude-agent-sdk==0.2.174",
        "yfinance>=1.7.0",
    )
    # Non-Python files are kept on purpose (Modal drops them by default): the
    # earnings-preview skill (preview_agent/workspace/.claude/...) and the model
    # and taxonomy artifacts (em_agent/artifacts/). em_agent loads the model with
    # `ml`; it does not need the research package suprise_dim.
    .add_local_python_source(
        "explaining_markets", "em_agent", "ml", "predict", "preview_agent", "shadow",
        ignore=lambda path: "__pycache__" in path.parts or path.suffix == ".pyc",
    )
)

# Distributed key-value store for idempotency, keyed on the Webhook-Id header.
# Three states:
#
#   "in_flight"   a job is running right now — skip duplicates so you never pay
#                 for the same model call twice
#   "done"        the API accepted a prediction — skip forever
#   absent        never seen, or the last attempt raised — (re)run it
#
# Marking an event done up front would be the bug: a failed prediction would
# look handled. This Dict persists across redeploys, so "done" is durable.
seen_webhooks = modal.Dict.from_name("em-webhook-dedupe", create_if_missing=True)

# Credentials are read from your local .env at deploy time (see .env.example).
# Prefer Modal's secret store instead? See docs/advanced.md.
secrets = [modal.Secret.from_dotenv(__file__)]

# Shadow mode: one folder per event on this Volume (own preview, competition
# preview, surprises, features, prediction), and atomic claims so each event's
# preview is generated once.
SHADOW_VOLUME = "em-shadow-data"
SHADOW_MOUNT = "/shadow"
shadow_volume = modal.Volume.from_name(SHADOW_VOLUME, create_if_missing=True)
preview_claims = modal.Dict.from_name("em-preview-claims", create_if_missing=True)
shadow_env = {"SHADOW_VOLUME": SHADOW_VOLUME}


def _claim(webhook_id):
    """Reserve this webhook_id. False means it's already in flight or done.

    `skip_if_exists` makes this an atomic claim, so two containers handling a
    duplicate delivery at the same moment can't both win.
    """
    if not webhook_id:
        return True
    return seen_webhooks.put(webhook_id, "in_flight", skip_if_exists=True)


async def _claim_aio(webhook_id):
    """`_claim` for the async route.

    Modal's blocking interfaces run their own event loop under the hood, so
    calling them from inside an `async def` stalls the loop — the exact problem
    ACKing first is meant to solve. The `.aio` variants are the async-native
    ones; the request path must use these, and only these.
    """
    if not webhook_id:
        return True
    return await seen_webhooks.put.aio(webhook_id, "in_flight", skip_if_exists=True)


def _release(webhook_id, submitted):
    """Mark the claim done on success, or drop it so a redelivery can retry."""
    if not webhook_id:
        return
    if submitted:
        seen_webhooks[webhook_id] = "done"
    else:
        seen_webhooks.pop(webhook_id, None)


@app.function(image=image, secrets=secrets, timeout=600, retries=0,
              volumes={SHADOW_MOUNT: shadow_volume}, env=shadow_env)
def predict_and_submit(event: dict, webhook_id: str | None = None):
    """Run the model and submit the prediction, off the request path.

    Runs in its own container, so it is unaffected by the web endpoint scaling
    down. The delivery has already been ACKed by the time this starts, which
    means nothing upstream will retry it — the single retry configured on the
    model call in predict.py is the only one you get.
    """
    from explaining_markets.client import submit_predictions
    from explaining_markets.config import Config
    from explaining_markets.event_utils import is_test, neutral_predictions
    from predict import predict

    submitted = False
    try:
        predictions = neutral_predictions(event) if is_test(event) else predict(event)
        submit_predictions(
            event_id=event["event_id"],
            predictions=predictions,
            config=Config.from_env(),
        )
        submitted = True
    except Exception as exc:
        # Log loudly — `modal app logs explaining-markets-starter` finds it.
        print(f"[ERROR] prediction failed for event {event.get('event_id')}: {exc}")
    finally:
        _release(webhook_id, submitted)


@app.function(image=image, secrets=secrets, schedule=modal.Period(minutes=15), timeout=300,
              volumes={SHADOW_MOUNT: shadow_volume}, env=shadow_env)
def schedule_previews():
    """Every 15 minutes: spawn a preview run for each event whose cutoff is under 12 hours away."""
    from preview_agent.events import fetch_events
    from shadow.schedule import calendar_window, claim, due_events
    from shadow.store import from_env

    store = from_env()
    store.refresh()
    start, end = calendar_window()
    due = due_events(fetch_events(start, end), store=store)
    spawned = 0
    for event in due:
        if claim(preview_claims, event["event_id"]):
            generate_own_preview.spawn(event)
            spawned += 1
    print(f"[shadow] {len(due)} events due, {spawned} preview runs started")


# At most MAX_PARALLEL_PREVIEWS (5) previews run at once; further spawns wait in Modal's queue.
@app.function(image=image, secrets=secrets, timeout=900, retries=0, max_containers=MAX_PARALLEL_PREVIEWS,
              volumes={SHADOW_MOUNT: shadow_volume}, env=shadow_env)
def generate_own_preview(event: dict):
    """Generate and store our own preview for one event (stopped at its knowledge cutoff)."""
    from preview_agent.generator import SKILL_PATH
    from shadow.runner import run_preview
    from shadow.schedule import release
    from shadow.store import from_env

    done = False
    try:
        if not SKILL_PATH.exists():
            raise RuntimeError(f"earnings-preview skill missing from the image: {SKILL_PATH}")
        record = run_preview(event, from_env())
        done = True
        print(f"[shadow] {event['event_id']}: {record.get('subtype')}, usable {record.get('usable')}, "
              f"cost ${record.get('total_cost_usd') or 0:.3f}")
    except Exception as exc:
        print(f"[ERROR] own preview failed for {event.get('event_id')}: {exc}")
    finally:
        # Free the claim on failure so a later check retries while the cutoff is still ahead.
        release(preview_claims, event["event_id"], done=done)


@app.function(image=image, secrets=secrets)
@modal.asgi_app(label="explaining-markets")
def web():
    from fastapi import FastAPI, Request, Response

    from explaining_markets import WebhookVerificationError, verify_webhook
    from explaining_markets.config import Config
    from explaining_markets.event_utils import log_deadline

    api = FastAPI(title="Explaining Markets starter")

    @api.get("/")
    def health() -> dict:
        return {"ok": True, "service": "explaining-markets-starter"}

    @api.post("/")
    @api.post("/competition/webhook")  # alias, so an explicit-path URL also works
    async def competition_webhook(request: Request) -> Response:
        config = Config.from_env()

        raw_body = await request.body()  # raw bytes — never request.json()
        try:
            event = verify_webhook(
                raw_body=raw_body,
                headers=request.headers,
                secret=config.webhook_secret,
            )
        except WebhookVerificationError as exc:
            return Response(content=str(exc), status_code=401)

        webhook_id = event.get("id")
        if not await _claim_aio(webhook_id):
            return Response(status_code=200)

        log_deadline(event)
        # Everything slow happens after this 200 goes out. The portal's "Test
        # Webhook" button sends a synthetic TEST event; it takes the same path
        # and submits a neutral prediction (accepted by the API, never scored)
        # so the test exercises your full receive -> submit loop.
        await predict_and_submit.spawn.aio(event, webhook_id)
        return Response(status_code=200)

    return api
