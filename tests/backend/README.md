# Backend 测试说明

本目录覆盖 `inference/infer_tool.py` 中的基础工具函数与 `Svc` 初始化流程，重点验证无需真实模型资源即可保证的行为。

## 测试模块
- `test_temp_utils.py`：缓存文件读写、过期清理逻辑。
- `test_array_utils.py`：数组填充与切片等纯函数。
- `test_format_wav.py`：`format_wav` 对非 WAV 输入的转换及早退出分支。
- `test_path_utils.py`：路径工具函数，包括递归搜集文件、目录创建与 MD5 计算。
- `test_svc_init.py`：在大量依赖被 mock 的前提下，验证 `Svc` 初始化时的关键属性设置。

## 运行方式
同样推荐使用 Conda 解释器，并设置 `SOVITS_WEBUI_HEADLESS=1` 以防其他测试构建 Gradio UI：

```bash
SOVITS_WEBUI_HEADLESS=1 /root/anaconda3/bin/pytest tests/backend -q
```

## Mock 策略
- 对文件系统、librosa、soundfile、torch 等重依赖采用 `pytest` 的 `monkeypatch` 进行替换。
- `test_svc_init.py` 中以轻量的 `DummySynth` 和伪造的 `hps` 对象模拟模型加载流程，确保不会触发真实权重加载或 GPU 操作。
