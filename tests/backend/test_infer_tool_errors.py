import numpy as np
import pytest
import torch

from inference import infer_tool


def build_stub_svc():
    svc = object.__new__(infer_tool.Svc)
    svc.spk2id = {"demo": 0}
    svc.dev = torch.device("cpu")
    svc.target_sample = 16000
    svc.hop_size = 320
    svc.feature_retrieval = False
    svc.cluster_model = None
    svc.dtype = torch.float32
    svc.volume_extractor = type("VE", (), {"extract": lambda self, x: torch.ones(1, x.shape[-1] // 2)})()
    svc.net_g_ms = type(
        "Net", (),
        {
            "infer": lambda self, *a, **k: (torch.zeros(1, 1, 10), torch.zeros(1, 10)),
            "to": lambda self, device: self,
        },
    )()
    svc.vocoder = type("VOC", (), {"extract": lambda self, x, sr: torch.zeros(1, 10), "infer": lambda self, mel, f0: torch.zeros(1, mel.shape[-1])})()
    class DummyHubert:
        def encoder(self, x):
            frames = max(1, x.shape[-1] // 400)
            return torch.ones(1, 2, max(frames, 1))

    svc.hubert_model = DummyHubert()

    def fake_repeat(content, target_len, mode):
        if not torch.is_tensor(content):
            content = torch.as_tensor(content)
        if content.dim() == 1:
            content = content.unsqueeze(0)
        return torch.ones(content.shape[0], target_len, dtype=content.dtype)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(infer_tool.utils, "repeat_expand_2d", fake_repeat)

    class DummyResample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, wav):
            return wav

        def to(self, device):
            return self

    svc.audio16k_resample_transform = DummyResample(16000, 16000)
    svc.audio_resample_transform = DummyResample(16000, 16000)
    svc.f0_predictor_object = type("F0", (), {"name": "pm", "compute_f0_uv": lambda self, wav: (np.ones(5), np.zeros(5))})()
    svc.unit_interpolate_mode = "left"
    return svc


def test_get_unit_f0_missing_feature_index(monkeypatch):
    svc = build_stub_svc()
    svc.feature_retrieval = True
    svc.cluster_model = {}

    with pytest.raises(KeyError):
        svc.get_unit_f0(
            wav=np.ones(1600, dtype=np.float32),
            tran=0,
            cluster_infer_ratio=0.5,
            speaker="demo",
            f0_filter=False,
            f0_predictor="pm",
            cr_threshold=0.05,
        )


def test_slice_inference_invalid_spk_mix(monkeypatch):
    svc = build_stub_svc()
    svc.spk2id = {"demo": 0, "user": 1}

    monkeypatch.setattr(
        infer_tool.slicer,
        "cut",
        lambda path, db_thresh=-30, min_len=5000: {"0": {"slice": False, "split_time": "0,800"}},
    )
    monkeypatch.setattr(
        infer_tool.slicer,
        "chunks2audio",
        lambda *_: ([(False, np.ones(1600))], 16000),
    )

    with pytest.raises(RuntimeError):
        svc.slice_inference(
            raw_audio_path="clip.wav",
            spk=[[ [0.0, 0.5, 0.0, 0.2] ], [ [0.5, 1.0, 0.2, -0.1] ]],
            tran=0,
            slice_db=-40,
            cluster_infer_ratio=0,
            auto_predict_f0=False,
            noice_scale=0.4,
            pad_seconds=0.5,
            clip_seconds=0,
            lg_num=0,
            lgr_num=0.75,
            f0_predictor="pm",
            enhancer_adaptive_key=0,
            cr_threshold=0.05,
            k_step=100,
            use_spk_mix=True,
            second_encoding=False,
            loudness_envelope_adjustment=1,
        )


def test_infer_missing_speaker(monkeypatch):
    svc = build_stub_svc()
    svc.spk2id = {"demo": 0}
    wav = torch.zeros(1, 1600)
    monkeypatch.setattr(infer_tool.torchaudio, "load", lambda *_: (wav, 16000))
    class DummyResampleCtor:
        def __init__(self, orig, new):
            self.orig = orig
            self.new = new

        def __call__(self, wav):
            return wav

    monkeypatch.setattr(
        infer_tool.torchaudio,
        "transforms",
        type("T", (), {"Resample": lambda orig, new: DummyResampleCtor(orig, new)}),
    )

    with pytest.raises(RuntimeError):
        svc.infer(
            speaker="unknown",
            tran=0,
            raw_path="clip.wav",
            cluster_infer_ratio=0,
            auto_predict_f0=False,
            noice_scale=0.4,
            f0_filter=False,
            f0_predictor="pm",
            enhancer_adaptive_key=0,
            cr_threshold=0.05,
            k_step=100,
            frame=0,
            spk_mix=False,
            second_encoding=False,
            loudness_envelope_adjustment=1,
        )
