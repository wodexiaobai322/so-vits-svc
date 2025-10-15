import json
from pathlib import Path

import numpy as np
import pytest
import gradio as gr


class DummyDropdown:
    def __init__(self):
        self.updated = None

    def update(self, **kwargs):
        self.updated = kwargs
        return kwargs


def test_mix_submit_click_success(monkeypatch, webui_module):
    captured = {}

    def fake_mix_model(paths, rates, mode):
        captured["paths"] = paths
        captured["rates"] = rates
        captured["mode"] = mode
        return "/tmp/output.pth"

    monkeypatch.setattr(webui_module, "mix_model", fake_mix_model)

    payload = json.dumps({"model_a.pth": 70, "model_b.pth": 30})
    message = webui_module.mix_submit_click(payload, "凸组合")

    assert "成功" in message
    assert captured["paths"] == ("model_a.pth", "model_b.pth")
    assert captured["rates"] == (70, 30)
    assert captured["mode"] == 0


def test_mix_submit_click_invalid_payload(webui_module):
    with pytest.raises(gr.Error):
        webui_module.mix_submit_click("not-json", "凸组合")


def test_model_analysis_local_model(monkeypatch, tmp_path, webui_module):
    # create local model directory
    model_dir = tmp_path / "demo_model"
    model_dir.mkdir()
    (model_dir / "voice.pth").write_bytes(b"fake")
    (model_dir / "config.json").write_text("{}")

    dummy_sid = DummyDropdown()
    monkeypatch.setattr(webui_module, "sid", dummy_sid)

    class DummySvc:
        def __init__(self, *args, **kwargs):
            self.spk2id = {"demo": 0}
            self.dev = "cpu"
            self.shallow_diffusion = False
            self.only_diffusion = False

        def unload_model(self):
            pass

    monkeypatch.setattr(webui_module, "Svc", DummySvc)

    sid_update, message = webui_module.modelAnalysis(
        model_path=None,
        config_path=None,
        cluster_model_path=None,
        device="cpu",
        enhance=False,
        diff_model_path=None,
        diff_config_path=None,
        only_diffusion=False,
        use_spk_mix=False,
        local_model_enabled=True,
        local_model_selection=str(model_dir),
    )

    assert sid_update["choices"] == ["demo"]
    assert sid_update["value"] == "demo"
    assert "成功加载模型" in message
    assert webui_module.model is not None


def test_model_unload_resets_model(monkeypatch, webui_module):
    dummy_sid = DummyDropdown()
    monkeypatch.setattr(webui_module, "sid", dummy_sid)

    class DummyModel:
        def __init__(self):
            self.unloaded = False

        def unload_model(self):
            self.unloaded = True

    dummy = DummyModel()
    webui_module.model = dummy
    monkeypatch.setattr(webui_module.torch.cuda, "empty_cache", lambda: None)

    sid_update, message = webui_module.modelUnload()

    assert dummy.unloaded
    assert sid_update["choices"] == []
    assert "卸载" in message
    assert webui_module.model is None


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
