import sys
import types

import numpy as np
import pytest
import torch


librosa_module = sys.modules["librosa"]
librosa_module.to_mono = lambda waveform: np.asarray(waveform)
librosa_module.feature = types.SimpleNamespace(
    rms=lambda y, frame_length, hop_length: np.abs(np.linspace(0.5, 0.1, int(len(y) / hop_length) + 1, dtype=np.float32))[None, :]
)
librosa_module.load = lambda path, sr=None: (np.linspace(0, 1, 800, dtype=np.float32), 16000)


def fake_torchaudio_load(path):
    data = torch.linspace(-1, 1, steps=800).unsqueeze(0)
    return data, 16000

torchaudio_module = sys.modules["torchaudio"]
torchaudio_module.load = fake_torchaudio_load

from inference import slicer


def test_slicer_short_wave_returns_single_chunk():
    s = slicer.Slicer(sr=16000, threshold=-40.0, min_length=500, min_interval=300, hop_size=20, max_sil_kept=500)
    waveform = np.zeros(200, dtype=np.float32)
    result = s.slice(waveform)
    assert isinstance(result, dict)
    assert result == {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}


def test_cut_and_chunks2audio_workflow():
    chunks = slicer.cut("dummy.wav", db_thresh=-35, min_len=5000)
    assert isinstance(chunks, dict)
    audio_segments, sr = slicer.chunks2audio("dummy.wav", chunks)
    assert sr == 16000
    assert isinstance(audio_segments, list)
    assert len(audio_segments) > 0
    for flag, segment in audio_segments:
        assert isinstance(flag, bool)
        assert isinstance(segment, np.ndarray)
        assert segment.ndim == 1


def test_slicer_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        slicer.Slicer(sr=16000, threshold=-20, min_length=100, min_interval=40, hop_size=80, max_sil_kept=200)


def test_slicer_handles_extended_silence(monkeypatch):
    test_slicer = slicer.Slicer(
        sr=100,
        threshold=-20,
        min_length=50,
        min_interval=30,
        hop_size=10,
        max_sil_kept=20,
    )
    waveform = np.linspace(0, 1, num=200, dtype=np.float32)

    rms_values = np.array([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6]], dtype=np.float32)
    monkeypatch.setattr(librosa_module.feature, "rms", lambda *a, **k: rms_values)

    chunks = test_slicer.slice(waveform)

    silent_starts = [
        int(entry["split_time"].split(",")[0]) for entry in chunks.values() if entry["slice"]
    ]
    assert silent_starts[0] == 0
    assert silent_starts[1] == test_slicer.hop_size * 4
    assert any(not entry["slice"] for entry in chunks.values())


def test_chunks2audio_downmixes_stereo(monkeypatch):
    stereo = torch.stack(
        [
            torch.linspace(0, 1, steps=6, dtype=torch.float32),
            torch.linspace(1, 2, steps=6, dtype=torch.float32),
        ]
    )
    monkeypatch.setattr(
        slicer.torchaudio,
        "load",
        lambda *_: (stereo, 22050),
    )
    chunks = {"0": {"slice": False, "split_time": "0,3"}, "1": {"slice": True, "split_time": "3,6"}}
    segments, sr = slicer.chunks2audio("dummy.wav", chunks)

    assert sr == 22050
    assert len(segments) == 2
    assert np.allclose(segments[0][1], np.mean(stereo.numpy(), axis=0)[:3])
    assert segments[1][0] is True
