import sys
import types
from types import SimpleNamespace

import numpy as np
import torch

import pytest


librosa_stub = types.ModuleType("librosa")
librosa_stub.load = lambda path, sr=None: (np.zeros(1), 16000)
librosa_stub.to_mono = lambda wav: np.asarray(wav)
librosa_stub.feature = SimpleNamespace(rms=lambda y, frame_length, hop_length: np.zeros((1, 1)))
sys.modules.setdefault("librosa", librosa_stub)

cluster_stub = types.ModuleType("cluster")
cluster_stub.get_cluster_model = lambda path: None
cluster_stub.get_cluster_center_result = lambda model, data, speaker: np.zeros_like(data)
sys.modules.setdefault("cluster", cluster_stub)

sklearn_stub = types.ModuleType("sklearn")
sklearn_cluster_stub = types.ModuleType("sklearn.cluster")
sklearn_cluster_stub.MiniBatchKMeans = lambda *a, **k: None
sklearn_stub.cluster = sklearn_cluster_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.cluster", sklearn_cluster_stub)

diffusion_stub = types.ModuleType("diffusion")
diffusion_stub.unit2mel = types.ModuleType("diffusion.unit2mel")
diffusion_stub.unit2mel.load_model_vocoder = lambda *a, **k: (SimpleNamespace(), SimpleNamespace(infer=lambda mel, f0: torch.zeros(1, 1)), SimpleNamespace(data=SimpleNamespace(sampling_rate=16000, block_size=320, encoder="dummy"), infer=SimpleNamespace(speedup=1, method="default")))
sys.modules.setdefault("diffusion", diffusion_stub)
sys.modules.setdefault("diffusion.unit2mel", diffusion_stub.unit2mel)

from inference import infer_tool


class IdentityResample:
    def __call__(self, wav):
        return wav

    def to(self, device):
        return self


def build_stub_svc():
    svc = object.__new__(infer_tool.Svc)
    svc.target_sample = 16000
    svc.hop_size = 320
    svc.dev = torch.device("cpu")
    svc.unit_interpolate_mode = "left"
    svc.feature_retrieval = False
    svc.big_npy = None
    svc.now_spk_id = -1
    svc.spk2id = {"demo": 0}
    svc.hubert_model = SimpleNamespace(encoder=lambda wav: torch.ones(1, max(1, wav.shape[-1] // 400)))
    svc.audio16k_resample_transform = IdentityResample()
    return svc


def test_get_unit_f0_raises_on_empty(monkeypatch):
    svc = build_stub_svc()

    class ZeroF0:
        name = "pm"

        def compute_f0_uv(self, wav):
            length = max(len(wav) // 400, 5)
            return np.zeros(length), np.zeros(length)

    svc.f0_predictor_object = ZeroF0()
    wav = np.zeros(1600, dtype=np.float32)

    with pytest.raises(infer_tool.F0FilterException):
        svc.get_unit_f0(wav, 0, 0, "demo", True, "pm")


def test_get_unit_f0_feature_retrieval(monkeypatch):
    svc = build_stub_svc()

    class DummyF0:
        name = "pm"

        def compute_f0_uv(self, wav):
            length = max(len(wav) // 400, 5)
            return np.ones(length), np.zeros(length)

    class DummyIndex:
        ntotal = 1

        def reconstruct_n(self, start, n):
            return np.ones((n, 4))

        def search(self, feat_np, k):
            score = np.ones((feat_np.shape[0], k))
            ix = np.zeros((feat_np.shape[0], k), dtype=int)
            return score, ix

    monkeypatch.setattr(
        infer_tool.utils,
        "repeat_expand_2d",
        lambda content, target_len, mode: torch.ones(1, target_len),
    )

    svc.feature_retrieval = True
    svc.f0_predictor_object = DummyF0()
    svc.cluster_model = {0: DummyIndex()}
    svc.spk2id = {"demo": 0}

    wav = np.ones(1600, dtype=np.float32)
    c, f0, uv = svc.get_unit_f0(wav, 0, 0.5, "demo", False, "pm")

    assert c.shape[-1] == f0.shape[1]
    assert svc.big_npy is not None
    assert svc.now_spk_id == 0
