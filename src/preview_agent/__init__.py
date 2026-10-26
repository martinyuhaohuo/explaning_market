"""Generate our own earnings previews with the Claude Agent SDK, the way the competition does.

The competition runs the earnings-preview skill from Anthropic's
earnings-reviewer plugin in the Claude Code harness, with a one-line request
naming the company, ticker, fiscal period and scheduled event time. When an
event arrives without a preview, a preview generated here before the event's
knowledge cutoff can stand in for it.

- `events`: the competition's events calendar (`GET /v1/events`).
- `names`: company names for tickers (yfinance, cached), since the calendar has none.
- `generator`: builds the request and runs one preview, with the knowledge
  cutoff enforced, saving the preview and a JSON record of the run.

The skill lives in `workspace/.claude/skills/earnings-preview/SKILL.md`; the
agent runs with `workspace/` as its working directory.
"""
