---
name: windows-text-editing-pitfalls
description: "How to patch text files safely in the solar repo on Windows (heredoc backslashes, CRLF from write_text, cp1252 stdout)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d327e5e5-b1d8-472a-8bda-4cf94ddd836e
  modified: 2026-10-25T08:14:29.229Z
---

When patching files in `C:\Users\baraky\Desktop\solar` from scripts:
- Python patch code inside a Bash heredoc can silently fail when it contains backslashes or
  quotes (an anchor stops matching). Write the patch script to the scratchpad with the Write tool
  and run it, or use the Edit tool.
- `Path.write_text()` on Windows writes CRLF. Use `newline="\n"` or `write_bytes`; the repo's
  `.gitattributes` wants LF for .md/.py/.yaml.
- Console output is cp1252: set `PYTHONIOENCODING=utf-8` when printing Hebrew, `−`, `✓`.
- Never use PowerShell Get-Content/Set-Content on PROJECT_MASTER.md (see CLAUDE.md).
- `Path.write_text(..., newline=<invalid>)` truncates the file **before** it raises. On
  2026-10-25 this emptied PROJECT_MASTER.md, which was restored with `git restore`. Never build
  escape sequences (`"\\n"`) by text substitution in patch scripts. After any patch, check the
  line count against `git show HEAD:<file> | wc -l`.

**Why:** in stage 6 (2026-10-20), a heredoc patch to decide_k.py did not apply, which caused a
wasted run (R6-03b in docs/run_journal.md), and write_text turned nine docs into CRLF files.

**How to apply:** for any multi-line text change, use Edit or a Write-tool patch script with
explicit asserts that each anchor occurs exactly once.
