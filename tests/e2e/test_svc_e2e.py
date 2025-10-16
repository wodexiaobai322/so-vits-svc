import importlib
import os
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.utils\.tensorboard")

pytestmark = pytest.mark.filterwarnings(
    "ignore:CUDA initialization:UserWarning",
    "ignore:distutils Version classes are deprecated.*:DeprecationWarning",
)


REQUIRED_PATHS = [
    Path("logs/44k/G_0.pth"),
    Path("configs/config.json"),
    Path("pretrain/checkpoint_best_legacy_500.pt"),
    Path("tests/fixtures/e2e/demo_input.wav"),
]


def _reload_real_module(name: str):
    """Reload a module that may have been replaced by the lightweight test stubs."""
    if name in sys.modules and getattr(sys.modules[name], "_sovits_stub", False):
        del sys.modules[name]
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        pass


def _ensure_real_dependencies():
    modules = [
        "torchaudio",
        "torchaudio.transforms",
        "diffusion",
        "diffusion.unit2mel",
    ]
    if "torchaudio.pipelines" not in sys.modules:
        pipelines_stub = types.ModuleType("torchaudio.pipelines")
        pipelines_stub._sovits_stub = True  # mark for potential cleanup
        sys.modules["torchaudio.pipelines"] = pipelines_stub
    for module_name in modules:
        _reload_real_module(module_name)


def _stub_fairseq_pdb():
    if "fairseq.pdb" not in sys.modules:
        dummy = types.ModuleType("fairseq.pdb")
        dummy.set_trace = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.is_debugging = lambda: False  # type: ignore[attr-defined]
        dummy.register_debugger = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.unregister_debugger = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.debug_handler = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["fairseq.pdb"] = dummy


@pytest.fixture(scope="session")
def real_svc(tmp_path_factory):
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")

    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("TORCH_LOGS", "off")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "CRITICAL")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    if "TENSORBOARD_BINARY" not in os.environ:
        os.environ["TENSORBOARD_BINARY"] = "False"
    _ensure_real_dependencies()
    _stub_fairseq_pdb()

    os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(tmp_path_factory.mktemp("numba_cache")))

    from inference import infer_tool

    svc = infer_tool.Svc(
        net_g_path="logs/44k/G_0.pth",
        config_path="configs/config.json",
        cluster_model_path="logs/44k/kmeans_10000.pt",
        diffusion_model_path="logs/44k/diffusion/model_0.pt",
        diffusion_config_path="configs/diffusion.yaml",
        nsf_hifigan_enhance=False,
        shallow_diffusion=False,
        only_diffusion=False,
        spk_mix_enable=False,
        feature_retrieval=False,
    )
    try:
        yield svc
    finally:
        svc.clear_empty()
        svc.unload_model()


@pytest.mark.e2e
def test_real_svc_infer_pipeline(real_svc, tmp_path):
    audio_path = Path("tests/fixtures/e2e/demo_input.wav")
    audio, sample_len, frame_count = real_svc.infer(
        speaker="nyaru",
        tran=0,
        raw_path=str(audio_path),
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noice_scale=0.4,
        f0_filter=False,
        f0_predictor="pm",
    )

    assert sample_len == audio.shape[-1]
    assert frame_count > 0
    assert float(torch_abs := audio.abs().max()) > 0.0, f"输出振幅为零: {torch_abs}"

    output_path = tmp_path / "svc_output.wav"
    sf.write(output_path, audio.cpu().numpy(), real_svc.target_sample)
    assert output_path.exists()


@pytest.mark.e2e
def test_webui_vc_fn_with_real_model(real_svc, tmp_path, monkeypatch):
    import webUI

    webUI.model = real_svc
    raw_dir = Path("raw")
    raw_dir.mkdir(exist_ok=True)

    input_wav = Path("tests/fixtures/e2e/demo_input.wav")
    message, output_path = webUI.vc_fn(
        sid="nyaru",
        input_audio=str(input_wav),
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )

    assert message == "Success"
    assert output_path and output_path.endswith(".wav")
    data, sr = sf.read(output_path)
    assert sr == real_svc.target_sample
    assert np.max(np.abs(data)) > 0.0

    tmp_copy = tmp_path / Path(output_path).name
    sf.write(tmp_copy, data, sr)
    assert tmp_copy.exists()

    Path(output_path).unlink(missing_ok=True)
