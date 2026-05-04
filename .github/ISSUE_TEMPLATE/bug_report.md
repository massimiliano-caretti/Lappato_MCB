---
name: Bug report
about: Report incorrect behaviour, crash, or unexpected output
title: "[bug] "
labels: bug
assignees: ''
---

## Summary

<!-- One sentence describing what went wrong. -->

## Environment

- LAPPATO_MCB version: `python -c "import lappato_mcb; print(lappato_mcb.__version__)"`
- Python version: `python --version`
- OS:
- Online or offline mode (`offline=True/False`):

## Reproducer

Minimal code that triggers the bug:

```python
from pathlib import Path
from lappato_mcb import LAPPATO_MCB
# ...
```

If a manifest is involved, paste the exact dict (or attach the file).

## Expected behaviour

<!-- What you thought would happen. -->

## Actual behaviour

<!-- What did happen. Include any traceback verbatim. -->

```
<stack trace here>
```

## Relevant artefacts

- [ ] `checkpoints/<run_tag>_lappato_mcb_log.md`
- [ ] `checkpoints/<run_tag>_lappato_mcb_meta.csv`
- [ ] `checkpoints/<run_tag>_lappato_mcb_weakness_cards.jsonl`

(Attach the smallest snippets that demonstrate the bug.)
