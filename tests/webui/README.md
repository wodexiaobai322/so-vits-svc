# WebUI 测试说明

该目录包含针对 `webUI.py` 推理端功能的轻量级单元测试，目标是在不依赖真实模型权重、GPU 或外部服务的前提下，对核心逻辑进行验证。

## 测试分组
- `test_mix_tools.py`：混音相关工具（静态声线融合 UI）的输入校验与状态更新。
- `test_model_management.py`：模型加载/卸载流程，包括本地模型扫描、上传模式及资源释放。
- `test_infer_api.py`：推理主流程 `vc_infer`、`vc_fn`、`vc_fn2` 的关键路径（音频预处理、参数校验、TTS 管线）。
- `test_utilities.py`：辅助函数（文本清理、Debug 切换、本地模型刷新）的行为。

## 运行方式
建议在 `SOVITS_WEBUI_HEADLESS=1` 下使用 Conda Python，避免导入 `webUI.py` 时构建完整 Gradio 界面：

```bash
SOVITS_WEBUI_HEADLESS=1 /root/anaconda3/bin/pytest tests/webui -q
```

## Mock 策略
- 使用 `pytest` 的 `monkeypatch` fixture 替换文件读写、子进程调用、Gradio 组件更新等副作用。
- 利用轻量数据（如 `numpy` 数组）模拟音频，确保推理函数仅走核心路径。
- 自动 fixture `reset_webui_globals` 在每个用例前后重置 `webUI` 模块的全局状态，避免污染。

## 扩展建议
- 若新增推理特性，可在相应文件继续补充测试，或按功能再拆分子模块。
- 对于依赖真实模型的场景，建议增加标记并默认跳过，需要时再在具备资源的环境手动运行。
