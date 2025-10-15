import json

import gradio as gr
import pytest


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


def test_upload_mix_append_file_combines_existing(webui_module):
    class DummyComponent:
        def update(self, **kwargs):
            return kwargs

    webui_module.mix_model_output1 = DummyComponent()

    class DummyFile:
        def __init__(self, name):
            self.name = name

    files = [DummyFile("a.pth")]
    sfiles = [DummyFile("b.pth")]

    paths, update = webui_module.upload_mix_append_file(files, sfiles)

    assert paths == ["a.pth", "b.pth"]
    assert json.loads(update["value"]) == {"a.pth": 100, "b.pth": 100}


def test_updata_mix_info_handles_none(webui_module):
    class DummyComponent:
        def update(self, **kwargs):
            return kwargs

    webui_module.mix_model_output1 = DummyComponent()
    update = webui_module.updata_mix_info(None)
    assert update["value"] == ""


def test_updata_mix_info_populates_files(webui_module):
    class DummyComponent:
        def update(self, **kwargs):
            return kwargs

    webui_module.mix_model_output1 = DummyComponent()

    class DummyFile:
        def __init__(self, name):
            self.name = name

    update = webui_module.updata_mix_info([DummyFile("x.pth")])
    assert json.loads(update["value"]) == {"x.pth": 100}
