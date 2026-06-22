---
name: feedback-no-panel-labels-in-plots
description: Never include (a), (b), (c) etc. panel labels inside matplotlib figures — always use LaTeX subcaption instead
metadata:
  type: feedback
---

Never bake panel labels like (a), (b), (c) into matplotlib figures (no `fig.text`, `ax.text`, or title with these labels).

**Why:** Labels belong in LaTeX as `\subcaption{}` so they are typeset consistently with the rest of the document and can be referenced with `\ref`.

**How to apply:** Any time a multi-panel figure is created or edited, omit panel labels from the Python plotting code entirely. Add `\subcaption{}` (with no text, or just the label) in the LaTeX `subfigure` environment instead.
