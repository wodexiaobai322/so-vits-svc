import sys
import types

import numpy as np
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
