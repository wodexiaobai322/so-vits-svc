import pytest
import importlib
import os


@pytest.fixture(scope="session")
def webui_module():
    os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")
    webui = importlib.import_module("webUI")
    return webui


@pytest.fixture(autouse=True)
def reset_webui_globals(webui_module):
    webui_module.model = None
    webui_module.debug = False
    yield
    webui_module.model = None
    webui_module.debug = False
