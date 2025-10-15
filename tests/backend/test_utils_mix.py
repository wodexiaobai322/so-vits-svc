import os
import sys
import types

import numpy as np
import torch

def _stub_dependencies():
    if "librosa" not in sys.modules:
        def rms(y, frame_length=None, hop_length=None):
            y = np.asarray(y)
            value = np.mean(np.abs(y))
            return np.array([[value]], dtype=np.float32)

        librosa_stub = types.ModuleType("librosa")
        librosa_stub.feature = types.SimpleNamespace(rms=rms)
        sys.modules["librosa"] = librosa_stub

    if "sklearn" not in sys.modules:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_cluster_stub = types.ModuleType("sklearn.cluster")
        sklearn_cluster_stub.MiniBatchKMeans = lambda *a, **k: None
        sklearn_stub.cluster = sklearn_cluster_stub
        sys.modules["sklearn"] = sklearn_stub
        sys.modules["sklearn.cluster"] = sklearn_cluster_stub

    if "faiss" not in sys.modules:
        sys.modules["faiss"] = types.ModuleType("faiss")


_stub_dependencies()

import utils


def test_mix_model_convex(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt1 = tmp_path / "m1.pth"
    ckpt2 = tmp_path / "m2.pth"

    template1 = {"model": {"weight": torch.tensor([1.0, 2.0])}}
    template2 = {"model": {"weight": torch.tensor([3.0, 4.0])}}
    torch.save(template1, ckpt1)
    torch.save(template2, ckpt2)

    output_path = utils.mix_model([str(ckpt1), str(ckpt2)], [50, 50], mode=0)
    mixed = torch.load(output_path)
    expected = (template1["model"]["weight"] + template2["model"]["weight"]) / 2
    torch.testing.assert_close(mixed["model"]["weight"], expected)


def test_mix_model_linear(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ckpt1 = tmp_path / "m1.pth"
    ckpt2 = tmp_path / "m2.pth"
    torch.save({"model": {"w": torch.tensor([1.0])}}, ckpt1)
    torch.save({"model": {"w": torch.tensor([3.0])}}, ckpt2)

    output_path = utils.mix_model([str(ckpt1), str(ckpt2)], [25, 75], mode=1)
    mixed = torch.load(output_path)
    expected = torch.tensor([1.0 * 0.25 + 3.0 * 0.75])
    torch.testing.assert_close(mixed["model"]["w"], expected)


def test_change_rms_scales(monkeypatch):
    data1 = np.ones(1600, dtype=np.float32) * 0.5
    data2 = torch.ones(1600, dtype=torch.float32)
    scaled = utils.change_rms(data1, 16000, data2.clone(), 16000, rate=0)
    assert isinstance(scaled, torch.Tensor)
    assert scaled.shape == data2.shape
    assert not torch.isnan(scaled).any()
