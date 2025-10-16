import contextlib
import io
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_inference_cli(tmp_path, base_required_paths, audio_fixture_path, svc_factory):
    missing = [str(path) for path in base_required_paths if not path.exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")

    svc = svc_factory()
    if getattr(svc, "net_g_ms", None) is not None:
        svc.unload_model()
    svc.clear_empty()

    raw_dir = Path("raw")
    results_dir = Path("results")
    raw_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    input_name = "demo_cli.wav"
    input_path = raw_dir / input_name
    shutil.copyfile(audio_fixture_path, input_path)

    if "fairseq.pdb" not in sys.modules:
        import types

        pdb_stub = types.ModuleType("fairseq.pdb")
        pdb_stub.set_trace = lambda *a, **k: None
        pdb_stub.is_debugging = lambda: False
        pdb_stub.register_debugger = lambda *a, **k: None
        pdb_stub.unregister_debugger = lambda *a, **k: None
        pdb_stub.debug_handler = lambda *a, **k: None
        sys.modules["fairseq.pdb"] = pdb_stub

    cache_dir = tmp_path / "numba_cache"
    cache_dir.mkdir(exist_ok=True)

    env_overrides = {
        "SOVITS_WEBUI_HEADLESS": "1",
        "NUMBA_DISABLE_JIT": "1",
        "TENSORBOARD_BINARY": "False",
        "TORCH_LOGS": "off",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "NUMBA_CACHE_DIR": str(cache_dir),
    }
    for key, value in env_overrides.items():
        os.environ[key] = value

    import importlib

    if "inference_main" in sys.modules:
        importlib.reload(sys.modules["inference_main"])
    else:
        importlib.import_module("inference_main")

    from inference import infer_tool

    def _cli_fake_slice(self, raw_audio_path, spk, tran, **kwargs):
        audio, _, _ = self.infer(
            speaker=spk,
            tran=tran,
            raw_path=str(raw_audio_path),
            cluster_infer_ratio=kwargs.get("cluster_infer_ratio", 0),
            auto_predict_f0=kwargs.get("auto_predict_f0", False),
            noice_scale=kwargs.get("noice_scale", 0.4),
            f0_filter=False,
            f0_predictor="harvest",
            loudness_envelope_adjustment=kwargs.get("loudness_envelope_adjustment", 1),
        )
        return audio.detach().cpu().numpy()

    infer_tool.Svc.slice_inference = _cli_fake_slice

    from inference_main import main as inference_main

    original_argv = sys.argv[:]
    sys.argv = [
        "inference_main.py",
        "-m",
        "logs/44k/G_0.pth",
        "-c",
        "configs/config.json",
        "-n",
        input_name,
        "-s",
        "nyaru",
        "-t",
        "0",
        "-wf",
        "wav",
        "-f0p",
        "harvest",
    ]
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            inference_main()
    finally:
        sys.argv = original_argv

    generated_files = list(results_dir.glob("demo_cli.wav_*_nyaru_*.wav"))
    assert generated_files, f"未找到 CLI 生成的音频。stdout: {stdout_buffer.getvalue()} stderr: {stderr_buffer.getvalue()}"
    output_path = generated_files[0]
    data, sr = sf.read(output_path)
    assert sr == 44100
    assert np.max(np.abs(data)) > 0.0

    tmp_output = tmp_path / output_path.name
    sf.write(tmp_output, data, sr)

    input_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
