import json
import time
from pathlib import Path

import pytest

from inference import infer_tool


def test_read_temp_creates_file(tmp_path):
    temp_file = tmp_path / "cache.json"
    data = infer_tool.read_temp(str(temp_file))
    assert temp_file.exists()
    assert data == {}


def test_read_temp_invalid_json(tmp_path):
    temp_file = tmp_path / "cache.json"
    temp_file.write_text("not-json")
    data = infer_tool.read_temp(str(temp_file))
    assert data == {"info": "temp_dict"}


def test_read_temp_prunes_old_entries(tmp_path, monkeypatch):
    temp_file = tmp_path / "cache.json"
    payload = {
        "info": {"time": int(time.time())},
        "expired": {"time": 0},
        "recent": {"time": int(time.time())}
    }
    temp_file.write_text(json.dumps(payload))
    monkeypatch.setattr(infer_tool.os.path, "getsize", lambda _: 60 * 1024 * 1024)
    data = infer_tool.read_temp(str(temp_file))
    assert "expired" not in data
    assert "recent" in data


def test_write_temp(tmp_path):
    temp_file = tmp_path / "cache.json"
    infer_tool.write_temp(str(temp_file), {"foo": 1})
    assert json.loads(temp_file.read_text()) == {"foo": 1}
