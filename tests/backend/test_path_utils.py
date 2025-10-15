from pathlib import Path

from inference import infer_tool


def test_get_end_file_recurses(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file.txt").write_text("1")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "skip.dat").write_text("1")
    (tmp_path / "file.txt").write_text("1")

    result = infer_tool.get_end_file(str(tmp_path), ".txt")
    expected = {str(tmp_path / "a" / "file.txt"), str(tmp_path / "file.txt")}
    assert set(result) == expected


def test_mkdir_creates_missing(tmp_path):
    path = tmp_path / "nested"
    infer_tool.mkdir([str(path)])
    assert path.exists()


def test_get_md5_hash():
    digest = infer_tool.get_md5(b"hello")
    assert digest == "5d41402abc4b2a76b9719d911017c592"
