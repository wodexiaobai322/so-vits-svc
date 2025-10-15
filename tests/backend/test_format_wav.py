import numpy as np

from inference import infer_tool


def test_format_wav_converts_non_wav(monkeypatch, tmp_path):
    src = tmp_path / "sample.mp3"
    src.write_bytes(b"fake")

    audio = np.array([0.1, -0.1], dtype=np.float32)

    def fake_load(path, mono=True, sr=None):
        assert str(path) == str(src)
        assert mono is True and sr is None
        return audio, 22050

    captured = {}

    def fake_write(path, data, sr):
        captured["path"] = path
        captured["data"] = data
        captured["sr"] = sr

    monkeypatch.setattr(infer_tool.librosa, "load", fake_load)
    monkeypatch.setattr(infer_tool.soundfile, "write", fake_write)

    infer_tool.format_wav(str(src))

    assert captured["path"] == src.with_suffix(".wav")
    assert captured["sr"] == 22050
    assert np.array_equal(captured["data"], audio)


def test_format_wav_noop_for_wav(monkeypatch, tmp_path):
    src = tmp_path / "sample.wav"
    src.write_bytes(b"fake")

    called = {"load": False, "write": False}

    monkeypatch.setattr(infer_tool.librosa, "load", lambda *a, **k: called.update(load=True))
    monkeypatch.setattr(infer_tool.soundfile, "write", lambda *a, **k: called.update(write=True))

    infer_tool.format_wav(str(src))

    assert called == {"load": False, "write": False}
