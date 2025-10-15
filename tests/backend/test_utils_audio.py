import sys
import types

import numpy as np
import torch


librosa_stub = types.ModuleType("librosa")
librosa_stub.load = lambda path, sr=None: (np.zeros(1), 16000)
librosa_stub.to_mono = lambda wav: np.asarray(wav)
librosa_stub.feature = types.SimpleNamespace(rms=lambda y, frame_length, hop_length: np.zeros((1, 1)))
sys.modules.setdefault("librosa", librosa_stub)

faiss_stub = types.ModuleType("faiss")
sys.modules.setdefault("faiss", faiss_stub)

sklearn_stub = types.ModuleType("sklearn")
sklearn_cluster_stub = types.ModuleType("sklearn.cluster")
sklearn_cluster_stub.MiniBatchKMeans = lambda *a, **k: None
sklearn_stub.cluster = sklearn_cluster_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.cluster", sklearn_cluster_stub)

from utils import normalize_f0, f0_to_coarse, repeat_expand_2d


def test_normalize_f0_scales_with_mask():
    f0 = torch.ones(1, 1, 4) * 200
    x_mask = torch.ones(1, 1, 4)
    uv = torch.tensor([[1, 0, 1, 0]], dtype=torch.float).unsqueeze(1)
    result = normalize_f0(f0, x_mask, uv, random_scale=False)
    assert result.shape[-1] == f0.shape[-1]
    assert not torch.isnan(result).any()

    zero_uv = torch.zeros_like(uv)
    zero_result = normalize_f0(f0, x_mask, zero_uv, random_scale=False)
    zero_flat = zero_result.reshape(-1)
    assert torch.allclose(zero_flat[: f0.numel()], f0.reshape(-1))


def test_f0_to_coarse_bounds():
    coarse = f0_to_coarse(torch.tensor([[0.0, 1200.0]]))
    assert coarse.shape == (1, 2)
    assert coarse.min() >= 0
    assert coarse.max() <= 254


def test_repeat_expand_matches_length():
    content = torch.arange(6, dtype=torch.float).view(2, 3)
    expanded = repeat_expand_2d(content, 6, mode="left")
    assert expanded.shape == (2, 6)
    assert torch.allclose(expanded[:, 0], content[:, 0])
