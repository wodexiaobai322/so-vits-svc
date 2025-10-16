import importlib
import os
import sys
import types
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"^websockets\.legacy",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"^uvicorn\.protocols\.websockets",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"^gradio\.routes",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*torch\.utils\.tensorboard.*",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="distutils Version classes are deprecated. Use packaging.version instead.",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*torch\.cuda.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="CUDA initialization: Unexpected error.*",
)

MODEL_PATH = Path("logs/44k/G_0.pth")
CONFIG_PATH = Path("configs/config.json")
DIFFUSION_MODEL_PATH = Path("logs/44k/diffusion/model_0.pt")
DIFFUSION_CONFIG_PATH = Path("configs/diffusion.yaml")
SPEECH_ENCODER_PATH = Path("pretrain/checkpoint_best_legacy_500.pt")
NSF_HIFIGAN_PATH = Path("pretrain/nsf_hifigan/model")
AUDIO_FIXTURE_PATH = Path("tests/fixtures/e2e/demo_input.wav")

BASE_REQUIRED_PATHS = [
    MODEL_PATH,
    CONFIG_PATH,
    SPEECH_ENCODER_PATH,
    AUDIO_FIXTURE_PATH,
]

DIFFUSION_REQUIRED_PATHS = BASE_REQUIRED_PATHS + [
    DIFFUSION_MODEL_PATH,
    DIFFUSION_CONFIG_PATH,
]


def _reload_real_module(name: str):
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
        pipelines_stub._sovits_stub = True
        sys.modules["torchaudio.pipelines"] = pipelines_stub
    for module_name in modules:
        _reload_real_module(module_name)
    import torchaudio

    if getattr(torchaudio, "_sovits_stub", False):
        pytest.skip("torchaudio 未在当前环境中可用，跳过真实模型端到端测试")


def _stub_fairseq_pdb():
    if "fairseq.pdb" not in sys.modules:
        dummy = types.ModuleType("fairseq.pdb")
        dummy.set_trace = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.is_debugging = lambda: False  # type: ignore[attr-defined]
        dummy.register_debugger = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.unregister_debugger = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        dummy.debug_handler = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["fairseq.pdb"] = dummy


def _prepare_runtime(tmp_path_factory):
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")
    os.environ.setdefault("TORCH_LOGS", "off")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "CRITICAL")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    if "TENSORBOARD_BINARY" not in os.environ:
        os.environ["TENSORBOARD_BINARY"] = "False"
    cache_dir = tmp_path_factory.mktemp("numba_cache")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    _ensure_real_dependencies()
    _stub_fairseq_pdb()
    import importlib

    for module_name in [
        "inference.infer_tool",
        "inference.slicer",
    ]:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])


def create_svc(tmp_path_factory, **overrides):
    from inference import infer_tool

    _prepare_runtime(tmp_path_factory)
    defaults = dict(
        net_g_path=str(MODEL_PATH),
        config_path=str(CONFIG_PATH),
        cluster_model_path="",
        nsf_hifigan_enhance=False,
        diffusion_model_path=str(DIFFUSION_MODEL_PATH),
        diffusion_config_path=str(DIFFUSION_CONFIG_PATH),
        shallow_diffusion=False,
        only_diffusion=False,
        spk_mix_enable=False,
        feature_retrieval=False,
    )
    defaults.update(overrides)
    svc = infer_tool.Svc(**defaults)
    return svc


def _check_required(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")


@pytest.fixture(scope="session")
def audio_fixture_path():
    return AUDIO_FIXTURE_PATH


@pytest.fixture(scope="session")
def base_required_paths():
    return BASE_REQUIRED_PATHS


@pytest.fixture(scope="session")
def diffusion_required_paths():
    return DIFFUSION_REQUIRED_PATHS


@pytest.fixture(scope="session")
def svc_factory(tmp_path_factory):
    def _factory(**overrides):
        return create_svc(tmp_path_factory, **overrides)

    return _factory


@pytest.fixture(scope="session")
def real_svc(tmp_path_factory):
    _check_required(BASE_REQUIRED_PATHS + [DIFFUSION_MODEL_PATH, DIFFUSION_CONFIG_PATH])
    svc = create_svc(tmp_path_factory)
    try:
        yield svc
    finally:
        svc.clear_empty()
        svc.unload_model()
