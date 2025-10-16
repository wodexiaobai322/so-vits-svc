import os
import time

from inference import infer_tool
import utils


def test_clean_checkpoints_removes_old(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Ensure baseline checkpoints exist and older ones have smaller mtimes
    (model_dir / "G_0.pth").write_text("bootstrap")
    (model_dir / "D_0.pth").write_text("bootstrap")

    base_time = time.time()
    for idx in range(1, 5):
        g_path = model_dir / f"G_{idx}.pth"
        d_path = model_dir / f"D_{idx}.pth"
        g_path.write_text(f"G{idx}")
        d_path.write_text(f"D{idx}")
        # Stagger modification times so earlier indices appear older
        os.utime(g_path, (base_time + idx, base_time + idx))
        os.utime(d_path, (base_time + idx, base_time + idx))

    utils.clean_checkpoints(str(model_dir), n_ckpts_to_keep=2, sort_by_time=True)

    # Oldest beyond the two newest per prefix are removed
    assert not (model_dir / "G_1.pth").exists()
    assert not (model_dir / "G_2.pth").exists()
    assert not (model_dir / "D_1.pth").exists()
    assert not (model_dir / "D_2.pth").exists()

    # Recent models and initial bootstrap checkpoints remain
    assert (model_dir / "G_3.pth").exists()
    assert (model_dir / "G_4.pth").exists()
    assert (model_dir / "D_3.pth").exists()
    assert (model_dir / "D_4.pth").exists()
    assert (model_dir / "G_0.pth").exists()
    assert (model_dir / "D_0.pth").exists()


def test_latest_checkpoint_path_returns_highest(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create out-of-order checkpoints
    for idx in [1, 5, 3]:
        (model_dir / f"G_{idx}.pth").write_text("x")

    latest = utils.latest_checkpoint_path(str(model_dir))
    assert latest == str(model_dir / "G_5.pth")


def test_load_filepaths_and_text_parses_delimited(tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("a.wav|hello\nb.wav|world\n")

    entries = utils.load_filepaths_and_text(str(listing))
    assert entries == [["a.wav", "hello"], ["b.wav", "world"]]


def test_timeit_decorator(monkeypatch):
    times = iter([0.0, 0.1])
    monkeypatch.setattr(infer_tool.time, "time", lambda: next(times))
    logged = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: logged.append(args[0]))

    @infer_tool.timeit
    def sample(x):
        return x + 1

    result = sample(4)
    assert result == 5
    assert any("executing 'sample' costed" in entry for entry in logged)
