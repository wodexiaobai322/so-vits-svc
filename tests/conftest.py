import importlib
import os

import pytest


@pytest.fixture(scope="session")
def webui_module():
    os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")
    webui = importlib.import_module("webUI")
    return webui


@pytest.fixture(autouse=True)
def reset_webui_globals(webui_module):
    webui_module.model = None
    webui_module.debug = False
    webui_module.sid = None
    webui_module.mix_model_output1 = None
    webui_module.debug_button = None
    yield
    webui_module.model = None
    webui_module.debug = False
    webui_module.sid = None
    webui_module.mix_model_output1 = None
    webui_module.debug_button = None
