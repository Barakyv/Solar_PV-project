---
name: hebrew-rtl-display-rules
description: "Every reply to the user must be readable Hebrew RTL; English terms, IDs, numbers and paths go on separate lines"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d327e5e5-b1d8-472a-8bda-4cf94ddd836e
  modified: 2026-10-25T06:56:13.227Z
---

The user asked (2026-10-25, high priority) that every reply be fluent Hebrew with stable
right-to-left display. The rules:
- Write Hebrew sentences that read right to left without words jumping out of order.
- Avoid mixing Hebrew and English in one sentence when it can scramble the display.
- Put technical English terms, variable names, numbers, task IDs and file paths on their own
  line, preferably as a code span or list item.
- Never start a Hebrew sentence with an English identifier, a number or a technical abbreviation.
- Avoid lines that mix Hebrew, English, parentheses, slashes and dashes.
- Keep the natural RTL reading order in lists, headings and summaries as well.
- Use simple Markdown; avoid formatting that breaks direction (complex tables with mixed cells).
- If a mixed sentence cannot display stably, split it into two sentences or separate lines.
- Never sacrifice readability for brevity. Do not simplify the technical content; change how it
  is presented.
- Before each reply, check that the Hebrew will read in order despite English terms.

**Why:** mixed Hebrew–English lines render with words jumping around, so the user cannot read the
replies continuously.

**How to apply:** every chat reply to this user. Project documents in English (docs/*.md) are
unaffected; the Hebrew PROJECT_MASTER.md is a separate convention. Related:
[[windows-text-editing-pitfalls]].
