import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import flask_api
import flask_api_full_song


@pytest.fixture
def voice_client(monkeypatch):
    monkeypatch.setattr(flask_api, "raw_infer", True, raising=False)
    captured = {}

    class DummyModel:
        target_sample = 16000

        def __init__(self):
            self.calls = []

        def infer(self, speaker_id, pitch, wav_path, **kwargs):
            wav_path.seek(0)
            self.calls.append((speaker_id, pitch))
            return torch.zeros(16000), 16000

    dummy_model = DummyModel()
    monkeypatch.setattr(flask_api, "svc_model", dummy_model, raising=False)
    if hasattr(flask_api.torchaudio, "functional"):
        monkeypatch.setattr(
            flask_api.torchaudio.functional,
            "resample",
            lambda audio, orig, new: audio,
        )
    else:
        monkeypatch.setattr(
            flask_api.torchaudio,
            "functional",
            SimpleNamespace(resample=lambda audio, orig, new: audio),
            raising=False,
        )

    def fake_write(buffer, data, samplerate, format="wav"):
        captured["data"] = np.asarray(data)
        captured["samplerate"] = samplerate
        buffer.write(b"FAKE")
        buffer.seek(0)

    monkeypatch.setattr(flask_api.soundfile, "write", fake_write)

    return flask_api.app.test_client(), dummy_model, captured


def test_voice_change_model_raw_infer(monkeypatch, voice_client):
    client, dummy_model, captured = voice_client
    data = {
        "fPitchChange": "0",
        "sampleRate": "16000",
        "sSpeakId": "0",
        "sample": (io.BytesIO(b"dummy"), "input.wav"),
    }
    response = client.post("/voiceChangeModel", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.headers["Content-Type"] in {"audio/wav", "audio/x-wav"}
    assert dummy_model.calls == [(0, 0.0)]
    assert captured["samplerate"] == 16000


def test_voice_change_model_realtime(monkeypatch, voice_client):
    client, dummy_model, captured = voice_client
    monkeypatch.setattr(flask_api, "raw_infer", False, raising=False)

    class DummyRealTime:
        def __init__(self):
            self.calls = []

        def process(self, *args, **kwargs):
            self.calls.append(kwargs)
            return np.zeros(16000, dtype=np.float32)

    dummy_rt = DummyRealTime()
    monkeypatch.setattr(flask_api, "svc", dummy_rt, raising=False)

    data = {
        "fPitchChange": "1",
        "sampleRate": "8000",
        "sSpeakId": "1",
        "sample": (io.BytesIO(b"dummy"), "input.wav"),
    }
    response = client.post("/voiceChangeModel", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    assert dummy_rt.calls
    assert captured["samplerate"] == 8000


@pytest.fixture
def full_song_client(monkeypatch):
    captured = {}

    def fake_format(path):
        captured["format_path"] = path

    monkeypatch.setattr(flask_api_full_song.infer_tool, "format_wav", fake_format)
    monkeypatch.setattr(
        flask_api_full_song.slicer,
        "cut",
        lambda path, db_thresh=-40: {"0": {"slice": False, "split_time": "0,800"}},
    )
    monkeypatch.setattr(
        flask_api_full_song.slicer,
        "chunks2audio",
        lambda path, chunks: ([(False, np.ones(1600))], 16000),
    )

    class DummySvc:
        target_sample = 16000

        def __init__(self):
            self.calls = 0

        def infer(self, spk, tran, raw_path):
            self.calls += 1
            return torch.zeros(16000), 16000

        def clear_empty(self):
            pass

    dummy_model = DummySvc()
    monkeypatch.setattr(flask_api_full_song, "svc_model", dummy_model, raising=False)

    def fake_write(buffer, data, samplerate, format="wav"):
        captured["output_rate"] = samplerate
        captured["output_format"] = format
        buffer.write(b"FAKE")
        buffer.seek(0)

    monkeypatch.setattr(flask_api_full_song.soundfile, "write", fake_write)

    return flask_api_full_song.app.test_client(), captured, dummy_model


def test_wav2wav_endpoint(full_song_client):
    client, captured, dummy_model = full_song_client
    data = {
        "audio_path": "song.wav",
        "tran": "2",
        "spk": "demo",
        "wav_format": "flac",
    }
    response = client.post("/wav2wav", data=data)

    assert response.status_code == 200
    assert dummy_model.calls == 1
    assert captured["output_rate"] == 16000
    assert captured["output_format"] == "flac"
