import importlib
import os
import sys
import types

import pytest
import torch
import numpy as np


def _ensure_stubs():
    if "librosa" not in sys.modules:
        def _rms(y, frame_length=None, hop_length=None):
            y = np.asarray(y)
            value = np.mean(np.abs(y)) if y.size else 0.0
            return np.array([[value]], dtype=np.float32)

        librosa_stub = types.ModuleType("librosa")
        librosa_stub.feature = types.SimpleNamespace(rms=_rms)
        librosa_stub.to_mono = lambda wav: np.asarray(wav)
        librosa_stub.load = lambda path, sr=None: (np.zeros(1600, dtype=np.float32), 16000)
        librosa_stub.resample = lambda data, orig_sr, target_sr: np.asarray(data)
        sys.modules["librosa"] = librosa_stub

    if "sklearn" not in sys.modules:
        class _DummyKMeans:
            def __init__(self, *args, **kwargs):
                pass

            def fit(self, *args, **kwargs):
                return self

            def predict(self, data):
                return [0] * len(data)

        sklearn_stub = types.ModuleType("sklearn")
        sklearn_cluster_stub = types.ModuleType("sklearn.cluster")
        sklearn_cluster_stub.KMeans = _DummyKMeans
        sklearn_cluster_stub.MiniBatchKMeans = _DummyKMeans
        sklearn_stub.cluster = sklearn_cluster_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.cluster"] = sklearn_cluster_stub

    if "faiss" not in sys.modules:
        sys.modules["faiss"] = types.ModuleType("faiss")

    if "torchaudio" not in sys.modules:
        torchaudio_stub = types.ModuleType("torchaudio")
        torchaudio_stub.load = lambda path: (torch.zeros(1, 1), 16000)
        transforms_module = types.ModuleType("torchaudio.transforms")

        class _Resample:
            def __init__(self, orig_freq, new_freq):
                self.orig_freq = orig_freq
                self.new_freq = new_freq

            def __call__(self, wav):
                return wav

            def to(self, device):
                return self

        transforms_module.Resample = _Resample
        torchaudio_stub.transforms = transforms_module
        torchaudio_stub.set_audio_backend = lambda *args, **kwargs: None
        sys.modules["torchaudio"] = torchaudio_stub
        sys.modules["torchaudio.transforms"] = transforms_module

    if "diffusion" not in sys.modules:
        diffusion_stub = types.ModuleType("diffusion")
        unit2mel_stub = types.ModuleType("diffusion.unit2mel")

        def _load_model_vocoder(*args, **kwargs):
            dummy_model = types.SimpleNamespace(
                infer=lambda *a, **k: torch.zeros(1, 1),
                init_spkmix=lambda *a, **k: None,
            )
            dummy_vocoder = types.SimpleNamespace(extract=lambda x, sr: torch.zeros_like(x), infer=lambda mel, f0: torch.zeros(1, mel.shape[-1] if hasattr(mel, "shape") else 1))
            dummy_cfg = types.SimpleNamespace(
                data=types.SimpleNamespace(sampling_rate=16000, block_size=320, encoder="dummy"),
                infer=types.SimpleNamespace(speedup=1, method="default"),
            )
            return dummy_model, dummy_vocoder, dummy_cfg

        unit2mel_stub.load_model_vocoder = _load_model_vocoder
        diffusion_stub.unit2mel = unit2mel_stub
        sys.modules["diffusion"] = diffusion_stub
        sys.modules["diffusion.unit2mel"] = unit2mel_stub

    if "matplotlib" not in sys.modules:
        matplotlib_stub = types.ModuleType("matplotlib")
        matplotlib_stub.use = lambda *a, **k: None
        matplotlib_stub.cbook = types.SimpleNamespace()
        matplotlib_stub.colors = types.SimpleNamespace(Colormap=object, is_color_like=lambda *_: True)
        matplotlib_stub._api = types.SimpleNamespace()
        matplotlib_stub.rcParams = {}
        sys.modules["matplotlib"] = matplotlib_stub


_ensure_stubs()


@pytest.fixture(scope="session")
def webui_module():
    os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")
    webui = importlib.import_module("webUI")
    return webui


@pytest.fixture(autouse=True)
def reset_webui_globals(webui_module):
    webui_module.model = None
    webui_module.debug = False
    webui_module.sid = None
    webui_module.mix_model_output1 = None
    webui_module.debug_button = None
    yield
    webui_module.model = None
    webui_module.debug = False
    webui_module.sid = None
    webui_module.mix_model_output1 = None
    webui_module.debug_button = None
