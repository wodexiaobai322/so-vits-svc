# 集成测试说明

此目录包含针对 `inference/infer_tool.Svc` 的轻量集成测试，重点验证推理管线在 mock 环境下的端到端行为。

## 用例概述
- `test_svc_infer_pipeline.py::test_svc_infer_pipeline`：通过大量 mock（torchaudio、SynthesizerTrn、utils 等）构造一个 `Svc` 实例，校验 `infer` 方法能完成音频加载、F0 预测、模型调用与返回值整理。
- `test_svc_infer_pipeline.py::test_slice_inference_calls_infer`：模拟切片流程，确保 `slice_inference` 会针对每个音频段调用 `infer` 并拼接输出。

## 运行方式
建议继续使用 Conda 解释器，并保持 `SOVITS_WEBUI_HEADLESS=1`，以避免其它测试初始化 Gradio UI：

```bash
SOVITS_WEBUI_HEADLESS=1 /root/anaconda3/bin/pytest tests/integration -q
```

## Mock 策略
- 使用 `monkeypatch` 替换 `utils` 中的模型加载、F0 预测、特征扩展及 RMS 调整逻辑。
- 以简化版本的 `SynthesizerTrn`、`torchaudio.load` 与 `Resample` 替换重计算模块，确保测试运行快速稳定。
- 对 `slicer.cut`/`chunks2audio`、`Svc.infer` 等函数进行定向 mock，使得 `slice_inference` 能在不依赖真实音频文件的情况下测试控制流程。

如需扩展其它集成场景（例如包含扩散或动态声线融合），可参考当前 mock 模式另建对应测试。 
