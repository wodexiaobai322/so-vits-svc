import pytest


class DummyDropdown:
    def __init__(self):
        self.updated = None

    def update(self, **kwargs):
        self.updated = kwargs
        return kwargs


def test_model_analysis_local_model(monkeypatch, tmp_path, webui_module):
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


def test_model_analysis_upload_mode(monkeypatch, tmp_path, webui_module):
    model_path = tmp_path / "voice.pth"
    config_path = tmp_path / "config.json"
    model_path.write_bytes(b"fake")
    config_path.write_text("{}")

    dummy_sid = DummyDropdown()
    monkeypatch.setattr(webui_module, "sid", dummy_sid)

    class DummyFile:
        def __init__(self, path):
            self.name = str(path)

    class DummySvc:
        def __init__(self, *args, **kwargs):
            self.spk2id = {"demo": 0}
            self.dev = "cpu"
            self.shallow_diffusion = False
            self.only_diffusion = False

        def unload_model(self):
            pass

    monkeypatch.setattr(webui_module, "Svc", DummySvc)

    file_obj = DummyFile(model_path)
    cfg_obj = DummyFile(config_path)
    sid_update, _ = webui_module.modelAnalysis(
        model_path=file_obj,
        config_path=cfg_obj,
        cluster_model_path=None,
        device="cpu",
        enhance=False,
        diff_model_path=None,
        diff_config_path=None,
        only_diffusion=False,
        use_spk_mix=False,
        local_model_enabled=False,
        local_model_selection="",
    )

    assert sid_update["choices"] == ["demo"]


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
