# Automated Testing

The repository now includes lightweight automated tests that exercise the inference utilities (`inference/`) and WebUI helper functions (`webUI.py`). The suite avoids heavy model downloads and GPU requirements by mocking the expensive dependencies.

## Prerequisites

Install the project dependencies listed in `requirements.txt`. No extra packages beyond `pytest` are required (bundled with the suite).

## Running the Tests

Execute the tests from the repository root:

```bash
pytest
```

All tests should pass on a CPU-only machine. Use `-k` or `-m` selectors to narrow the scope if you extend the suite with slower scenarios.

## Extending the Suite

- Add new tests under `tests/`, grouping related behaviours per module.
- Prefer fixtures and monkeypatching to isolate external side effects.
- For heavier inference checks (e.g., end-to-end conversions), guard them behind `pytest` markers so they can be skipped in CI by default.
