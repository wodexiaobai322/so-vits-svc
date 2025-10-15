import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
import pytest


# Provide lightweight librosa stub before importing inference modules
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

class DummyMiniBatchKMeans:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    def predict(self, data):
        return np.zeros(len(data))

sklearn_cluster_stub.MiniBatchKMeans = DummyMiniBatchKMeans
sklearn_stub.cluster = sklearn_cluster_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.cluster", sklearn_cluster_stub)

diffusion_stub = types.ModuleType("diffusion")
unit2mel_stub = types.ModuleType("diffusion.unit2mel")
unit2mel_stub.load_model_vocoder = lambda *a, **k: (
    SimpleNamespace(),
    SimpleNamespace(infer=lambda mel, f0: torch.zeros(1, mel.shape[-1] if hasattr(mel, "shape") else 1)),
    SimpleNamespace(
        data=SimpleNamespace(sampling_rate=16000, block_size=320, encoder="dummy_encoder"),
        infer=SimpleNamespace(speedup=1, method="default"),
    ),
)
diffusion_stub.unit2mel = unit2mel_stub
sys.modules.setdefault("diffusion", diffusion_stub)
sys.modules.setdefault("diffusion.unit2mel", unit2mel_stub)

from inference import infer_tool


@pytest.fixture
def svc_env(monkeypatch, tmp_path):
    # ---- Patch utils helpers ----
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    hps = SimpleNamespace(
        data=SimpleNamespace(
            sampling_rate=16000,
            hop_length=320,
            unit_interpolate_mode=None,
            filter_length=1024,
        ),
        train=SimpleNamespace(segment_size=6400),
        model=AttrDict(
            vol_embedding=False,
            speech_encoder="dummy",
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            gin_channels=256,
            ssl_dim=256,
            n_speakers=10,
            use_spk_mix=False,
            use_spectral_norm=False,
        ),
        spk={"demo": 0},
    )

    monkeypatch.setattr(infer_tool.utils, "get_hparams_from_file", lambda *_: hps)
    monkeypatch.setattr(infer_tool.utils, "load_checkpoint", lambda *a, **k: (None, None, None, 0))

    class DummySpeechEncoder:
        def encoder(self, wav):
            length = wav.shape[-1] // 400 or 1
            return torch.ones(1, length)

    monkeypatch.setattr(infer_tool.utils, "get_speech_encoder", lambda *a, **k: DummySpeechEncoder())

    class DummyVolume:
        def __init__(self, hop):
            self.hop = hop

        def extract(self, wav):
            return torch.ones(1, wav.shape[-1] // self.hop + 1)

    monkeypatch.setattr(infer_tool.utils, "Volume_Extractor", DummyVolume)

    def fake_repeat_expand(tensor, target_len, mode):
        tensor = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
        return torch.nn.functional.interpolate(tensor.unsqueeze(0), size=target_len, mode="nearest").squeeze(0)

    monkeypatch.setattr(infer_tool.utils, "repeat_expand_2d", fake_repeat_expand)
    monkeypatch.setattr(infer_tool.utils, "change_rms", lambda *args, **kwargs: torch.zeros_like(args[2]))

    class DummyF0:
        name = "pm"

        def compute_f0_uv(self, wav):
            length = max(len(wav) // 400, 5)
            return np.ones(length), np.zeros(length)

    monkeypatch.setattr(infer_tool.utils, "get_f0_predictor", lambda *a, **k: DummyF0())

    # ---- Patch torchaudio ----
    class DummyResample:
        def __init__(self, orig, new):
            self.orig_freq = orig
            self.new_freq = new

        def __call__(self, wav):
            return wav

        def to(self, device):
            return self

    monkeypatch.setattr(infer_tool.torchaudio, "set_audio_backend", lambda *_: None)
    monkeypatch.setattr(infer_tool.torchaudio, "load", lambda path: (torch.linspace(0, 1, steps=1600).unsqueeze(0), 16000))
    monkeypatch.setattr(infer_tool.torchaudio, "transforms", SimpleNamespace(Resample=lambda o, n: DummyResample(o, n)))

    # ---- Patch Synthesizer/vocoder ----
    class DummySynth(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.mix_enabled = False

        def parameters(self):
            return [self._param]

        def half(self):
            return self

        def eval(self):
            return self

        def to(self, device):
            return self

        def EnableCharacterMix(self, *_args, **_kwargs):
            self.mix_enabled = True
            return self

        def infer(self, c, f0, g, uv, predict_f0, noice_scale, vol=None):
            frames = c.shape[-1]
            audio = torch.ones(1, 1, frames * 2)
            return audio, f0

    monkeypatch.setattr(infer_tool, "SynthesizerTrn", DummySynth)

    dummy_vocoder = SimpleNamespace(extract=lambda audio, sr: torch.zeros_like(audio), infer=lambda mel, f0: torch.zeros(1, mel.shape[-1]))

    monkeypatch.setattr(infer_tool, "cluster", SimpleNamespace(get_cluster_center_result=lambda *a, **k: np.zeros((a[1].shape[0], a[1].shape[1]))))

    svc = infer_tool.Svc(
        net_g_path="model.pth",
        config_path="config.json",
        cluster_model_path=str(tmp_path / "missing.pkl"),
        diffusion_model_path=str(tmp_path / "diffusion.pth"),
        diffusion_config_path=str(tmp_path / "diffusion.yaml"),
        shallow_diffusion=False,
        only_diffusion=False,
    )
    svc.vocoder = dummy_vocoder
    return svc


def test_svc_infer_pipeline(tmp_path, svc_env, monkeypatch):
    svc = svc_env
    wav_path = tmp_path / "clip.wav"
    # create simple tensor saved via torch to mimic torchaudio load (unused due to monkeypatch)
    wav_path.write_bytes(b"fake")

    audio, sample_len, n_frames = svc.infer(
        speaker="demo",
        tran=0,
        raw_path=str(wav_path),
        cluster_infer_ratio=0,
        auto_predict_f0=False,
    )

    assert isinstance(audio, torch.Tensor)
    assert sample_len == audio.shape[-1]
    assert n_frames > 0
    assert audio.shape[-1] > 0


def test_slice_inference_calls_infer(tmp_path, monkeypatch, svc_env):
    svc = svc_env

    samples = torch.linspace(-1, 1, steps=1600)
    wav_path = tmp_path / "slice.wav"
    torch.save(samples, wav_path)

    def fake_cut(path, db_thresh=-30):
        return {"0": {"slice": False, "split_time": "0,800"}, "1": {"slice": False, "split_time": "800,1600"}}

    def fake_chunks2audio(path, chunks):
        arr = np.linspace(0, 1, num=1600)
        return [
            (False, arr[:800]),
            (False, arr[800:]),
        ], 16000

    calls = []

    def fake_infer(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.zeros(3200), 3200, 10

    monkeypatch.setattr(infer_tool.slicer, "cut", fake_cut)
    monkeypatch.setattr(infer_tool.slicer, "chunks2audio", fake_chunks2audio)
    monkeypatch.setattr(infer_tool.Svc, "infer", lambda self, *a, **k: fake_infer(*a, **k))

    output = svc.slice_inference(
        raw_audio_path=str(wav_path),
        spk="demo",
        tran=0,
        slice_db=-40,
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noice_scale=0.4,
        pad_seconds=0.0,
        clip_seconds=0.0,
    )

    assert isinstance(output, np.ndarray)
    assert len(calls) == 2
    assert output.shape[0] > 0
