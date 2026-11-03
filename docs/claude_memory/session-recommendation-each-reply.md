---
name: session-recommendation-each-reply
description: "End every reply with a recommendation: continue in this session or open a new one, and why"
metadata:
  type: feedback
---

The user asked (2026-11-03): from now on, at the end of every reply, recommend whether to
continue in the current session or start a new one, to keep the Claude project orderly, save
tokens and keep only relevant context.

**Why:** long sessions carry context that no longer matters (installs, inspections, old
diagnoses), which costs tokens and clutters the work; the project's documents already hold the
state a new session needs.

**How to apply:**
- Last thing in every reply: one short Hebrew line (per [[hebrew-rtl-display-rules]]) with the
  recommendation and a one-sentence reason.
- Recommend a **new session** when the next step is a new stage or phase whose context is
  already in `PROJECT_MASTER.md` §1, `docs/run_journal.md` or a stage plan, or when this session
  is long and its remaining context is irrelevant to what comes next.
- Recommend **continuing here** when the next step is short or builds on something found in this
  session that is not yet written down.
- Before recommending a new session, make sure this session's results are in the documents and
  committed, or say exactly what is not.
