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

### End-to-end checks

Real模型端到端测试位于 `tests/e2e/`，默认使用 `pytest.mark.e2e` 标记。运行前请确认以下资源已就绪：

- `logs/44k/G_0.pth`
- `logs/44k/D_0.pth`
- `logs/44k/diffusion/model_0.pt`
- `pretrain/checkpoint_best_legacy_500.pt`
- 示例音频 `tests/fixtures/e2e/demo_input.wav`

执行方式：

```bash
SOVITS_WEBUI_HEADLESS=1 pytest -m e2e tests/e2e
```

成功后会验证真实 `Svc` 推理与 `webUI.vc_fn` 调用链是否能够产出非空音频。

## Extending the Suite

- Add new tests under `tests/`, grouping related behaviours per module.
- Prefer fixtures and monkeypatching to isolate external side effects.
- For heavier inference checks (e.g., end-to-end conversions), guard them behind `pytest` markers so they can be skipped in CI by default.
