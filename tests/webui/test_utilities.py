def test_text_clear_removes_symbols(webui_module):
    result = webui_module.text_clear("Hello, (world)\n !")
    assert result == "Helloworld!"


def test_debug_change_reads_checkbox(monkeypatch, webui_module):
    class DummyCheckbox:
        value = True

    webui_module.debug_button = DummyCheckbox()
    webui_module.debug = False
    webui_module.debug_change()
    assert webui_module.debug is True


def test_scan_local_models_filters(tmp_path, monkeypatch, webui_module):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "model.pth").write_text("fake")
    (good_dir / "config.json").write_text("{}")

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "only.json").write_text("{}")

    monkeypatch.setattr(webui_module, "local_model_root", str(tmp_path))
    results = webui_module.scan_local_models()
    assert results == [str(good_dir)]


def test_local_model_refresh_returns_dropdown(monkeypatch, webui_module):
    monkeypatch.setattr(webui_module, "scan_local_models", lambda: ["path1", "path2"])
    monkeypatch.setattr(
        webui_module.gr, "Dropdown", type("Dropdown", (), {"update": staticmethod(lambda **kw: kw)})
    )
    update = webui_module.local_model_refresh_fn()
    assert update["choices"] == ["path1", "path2"]
