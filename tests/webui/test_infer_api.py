import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_vc_infer_builds_expected_path(monkeypatch, tmp_path, webui_module):
    monkeypatch.chdir(tmp_path)

    audio = np.zeros(10, dtype=np.float32)
    write_calls = {}

    class DummyModel:
        def __init__(self):
            self.shallow_diffusion = False
            self.only_diffusion = False
            self.target_sample = 44100
            self.slice_called = None
            self.cleared = False

        def slice_inference(self, *args):
            self.slice_called = args
            return audio

        def clear_empty(self):
            self.cleared = True

    webui_module.model = DummyModel()

    def fake_write(path, data, sample_rate, format):
        write_calls["path"] = Path(path)
        write_calls["data"] = data
        write_calls["rate"] = sample_rate
        write_calls["format"] = format

    monkeypatch.setattr(webui_module.soundfile, "write", fake_write)

    output_path = webui_module.vc_infer(
        output_format="wav",
        sid="demo",
        audio_path="input.wav",
        truncated_basename="sample",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )

    expected_file = Path("results") / "result_sample_demo_0key_sovits.wav"
    assert Path(output_path) == expected_file
    assert write_calls["path"] == expected_file
    np.testing.assert_array_equal(write_calls["data"], audio)
    assert write_calls["rate"] == 44100
    assert write_calls["format"] == "wav"
    assert webui_module.model.cleared


def test_vc_fn_requires_audio(webui_module):
    webui_module.model = object()
    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio=None,
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )
    assert msg == "You need to upload an audio"
    assert output is None


def test_vc_fn_requires_model(webui_module):
    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio="anything.wav",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )
    assert msg == "You need to upload an model"
    assert output is None


def test_vc_fn_cluster_ratio_guard(webui_module):
    webui_module.model = SimpleNamespace(cluster_model=None, feature_retrieval=False)
    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio="dummy.wav",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0.3,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )
    assert "cluster model" in msg
    assert output is None


def test_vc_fn_converts_audio_and_calls_infer(monkeypatch, tmp_path, webui_module):
    webui_module.model = SimpleNamespace(feature_retrieval=True)
    input_audio = str(tmp_path / "clipabcdefxxxx.wav")

    stereo_samples = np.array([[1000, -1000], [2000, -2000]], dtype=np.int16)

    def fake_read(path, *args, **kwargs):
        assert path == input_audio
        return stereo_samples.copy(), 48000

    mono_context = {}

    def fake_to_mono(arr):
        mono_context["input_shape"] = arr.shape
        return arr.mean(axis=0)

    write_calls = []

    def fake_write(path, data, samplerate, format=None, subtype=None):
        write_calls.append(
            {"path": path, "data": data.copy(), "sr": samplerate, "format": format, "subtype": subtype}
        )

    infer_calls = {}

    def fake_vc_infer(*args, **kwargs):
        infer_calls["args"] = args
        infer_calls["kwargs"] = kwargs
        return "results/output.wav"

    monkeypatch.setattr(webui_module.soundfile, "read", fake_read)
    monkeypatch.setattr(webui_module.librosa, "to_mono", fake_to_mono)
    monkeypatch.setattr(webui_module.soundfile, "write", fake_write)
    monkeypatch.setattr(webui_module, "vc_infer", fake_vc_infer)

    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio=input_audio,
        output_format="flac",
        vc_transform=2,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-30,
        noise_scale=0.2,
        pad_seconds=1.0,
        cl_num=0,
        lg_num=0,
        lgr_num=0.5,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=50,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0.2,
    )

    truncated = Path(input_audio).stem[:-6]
    assert msg == "Success"
    assert output == "results/output.wav"
    assert mono_context["input_shape"] == (2, 2)
    assert write_calls[0]["path"] == os.path.join("raw", f"{truncated}.wav")
    assert write_calls[0]["sr"] == 48000
    assert write_calls[0]["format"] == "wav"
    assert write_calls[0]["data"].dtype == np.float32
    assert np.max(np.abs(write_calls[0]["data"])) <= 1.0
    assert infer_calls["args"][1] == "demo"
    assert infer_calls["args"][2] == os.path.join("raw", f"{truncated}.wav")
    assert infer_calls["args"][3] == truncated


def test_vc_fn2_requires_model(webui_module):
    msg, output = webui_module.vc_fn2(
        _text="hello",
        _lang="Auto",
        _gender="男",
        _rate=0,
        _volume=0,
        sid="demo",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )
    assert msg == "You need to upload an model"
    assert output is None


def test_vc_fn2_cluster_ratio_guard(webui_module):
    webui_module.model = SimpleNamespace(cluster_model=None, feature_retrieval=False)
    msg, output = webui_module.vc_fn2(
        _text="hello",
        _lang="Auto",
        _gender="男",
        _rate=0,
        _volume=0,
        sid="demo",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0.6,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )
    assert "cluster model" in msg
    assert output is None


def test_vc_fn2_runs_tts_pipeline(monkeypatch, webui_module):
    webui_module.model = SimpleNamespace(cluster_model=object(), feature_retrieval=False)

    run_calls = []
    monkeypatch.setattr(webui_module.subprocess, "run", lambda cmd: run_calls.append(cmd))

    monkeypatch.setattr(webui_module.librosa, "load", lambda _: (np.ones(4, dtype=np.float32), 22050))
    monkeypatch.setattr(webui_module.librosa, "resample", lambda data, orig_sr, target_sr: data * 2)

    writes = []
    monkeypatch.setattr(
        webui_module.soundfile,
        "write",
        lambda path, data, samplerate, subtype=None, format=None: writes.append(
            {"path": path, "data": data.copy(), "sr": samplerate, "subtype": subtype}
        ),
    )
    monkeypatch.setattr(webui_module.os, "remove", lambda path: writes.append({"removed": path}))

    infer_calls = []
    monkeypatch.setattr(webui_module, "vc_infer", lambda *args, **kwargs: infer_calls.append((args, kwargs)) or "result.wav")

    msg, output = webui_module.vc_fn2(
        _text="你好",
        _lang="Auto",
        _gender="男",
        _rate=0.1,
        _volume=-0.2,
        sid="demo",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )

    assert msg == "Success"
    assert output == "result.wav"
    assert run_calls[0][0] == webui_module.sys.executable
    assert writes[0]["path"] == "tts.wav"
    assert writes[0]["sr"] == 44100
    assert writes[0]["subtype"] == "PCM_16"
    assert np.allclose(writes[0]["data"], np.ones(4) * 2)
    assert writes[1]["removed"] == "tts.wav"
    assert infer_calls[0][0][2] == "tts.wav"
