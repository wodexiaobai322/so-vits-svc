from types import SimpleNamespace

import numpy as np
import torch

from inference import infer_tool


class DummyDiffusionModel:
    def __init__(self):
        self.init_calls = []
        self.last_call = None

    def init_spkmix(self, count):
        self.init_calls.append(count)

    def __call__(
        self,
        c,
        f0,
        vol,
        spk_id,
        spk_mix_dict,
        gt_spec,
        infer,
        infer_speedup,
        method,
        k_step,
    ):
        frames = f0.shape[1]
        self.last_call = {
            "c_shape": tuple(c.shape),
            "f0_shape": tuple(f0.shape),
            "vol_shape": tuple(vol.shape),
            "spk_id_shape": tuple(spk_id.shape),
            "gt_spec": gt_spec,
            "infer": infer,
            "speedup": infer_speedup,
            "method": method,
            "k_step": k_step,
        }
        return torch.ones(1, frames, dtype=torch.float32)


class DummyVocoder:
    def __init__(self):
        self.calls = []

    def infer(self, audio_mel, f0):
        self.calls.append(
            {"mel_shape": tuple(audio_mel.shape), "f0_shape": tuple(f0.shape)}
        )
        length = audio_mel.shape[-1]
        return torch.ones(1, length, dtype=torch.float32)


class DummyVolumeExtractor:
    def __init__(self, hop_size):
        self.hop_size = max(1, hop_size)
        self.calls = []

    def extract(self, audio):
        frames = max(1, (audio.shape[-1] + self.hop_size - 1) // self.hop_size)
        result = torch.ones(audio.shape[0], frames, dtype=torch.float32)
        self.calls.append(frames)
        return result


class DummyEncoder:
    def encoder(self, wav):
        frames = max(1, wav.shape[-1] // 2)
        return torch.ones(1, frames, dtype=torch.float32)


class DummyF0Predictor:
    name = "pm"

    def compute_f0_uv(self, wav):
        length = max(1, len(wav) // 2)
        return np.ones(length, dtype=np.float32), np.zeros(length, dtype=np.float32)


class DummyResample:
    def __init__(self, orig, new):
        self.orig_freq = orig
        self.new_freq = new

    def __call__(self, wav):
        return wav

    def to(self, device=None):
        return self


def test_only_diffusion_infer_uses_vocoder(monkeypatch, tmp_path):
    diff_model_path = tmp_path / "diffusion_model.pt"
    diff_cfg_path = tmp_path / "diffusion.yaml"
    diff_model_path.write_text("stub")
    diff_cfg_path.write_text("stub")

    dummy_diffusion = DummyDiffusionModel()
    dummy_vocoder = DummyVocoder()
    diffusion_args = SimpleNamespace(
        data=SimpleNamespace(
            sampling_rate=8000,
            block_size=2,
            encoder="dummy",
            unit_interpolate_mode="left",
        ),
        infer=SimpleNamespace(speedup=2, method="fast"),
        spk={"demo": 0},
    )

    monkeypatch.setattr(
        infer_tool,
        "load_model_vocoder",
        lambda *a, **k: (dummy_diffusion, dummy_vocoder, diffusion_args),
    )
    monkeypatch.setattr(
        infer_tool.utils,
        "get_speech_encoder",
        lambda *a, **k: DummyEncoder(),
    )
    monkeypatch.setattr(
        infer_tool.utils,
        "Volume_Extractor",
        lambda hop: DummyVolumeExtractor(hop),
    )
    def fake_repeat(content, target_len, mode):
        if not torch.is_tensor(content):
            content = torch.as_tensor(content, dtype=torch.float32)
        if content.dim() == 1:
            content = content.unsqueeze(0)
        return torch.ones(content.shape[0], target_len, dtype=content.dtype)

    monkeypatch.setattr(infer_tool.utils, "repeat_expand_2d", fake_repeat)
    monkeypatch.setattr(
        infer_tool.utils,
        "get_f0_predictor",
        lambda *a, **k: DummyF0Predictor(),
    )
    torchaudio_stub = SimpleNamespace(
        set_audio_backend=lambda *_: None,
        load=lambda *_: (torch.linspace(-1, 1, steps=10).unsqueeze(0), 16000),
        transforms=SimpleNamespace(Resample=lambda orig, new: DummyResample(orig, new)),
    )
    monkeypatch.setattr(infer_tool, "torchaudio", torchaudio_stub)

    svc = infer_tool.Svc(
        net_g_path="model.pth",
        config_path=str(tmp_path / "config.json"),
        diffusion_model_path=str(diff_model_path),
        diffusion_config_path=str(diff_cfg_path),
        shallow_diffusion=False,
        only_diffusion=True,
        spk_mix_enable=True,
    )

    audio, sample_len, n_frames = svc.infer(
        speaker="demo",
        tran=0,
        raw_path="dummy.wav",
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noice_scale=0.2,
        k_step=8,
        loudness_envelope_adjustment=1,
    )

    assert torch.is_tensor(audio)
    assert sample_len == audio.shape[-1]
    assert n_frames > 0
    assert dummy_diffusion.init_calls == [1]
    assert dummy_diffusion.last_call["method"] == "fast"
    assert dummy_vocoder.calls[0]["mel_shape"][-1] == n_frames
