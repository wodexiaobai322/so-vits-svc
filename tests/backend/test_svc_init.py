from types import SimpleNamespace

import pytest

from inference import infer_tool


class DummySynth:
    def __init__(self, *args, **kwargs):
        self._params = [SimpleNamespace(dtype="float32")]
        self.half_called = False
        self.eval_called = False
        self.device = None

    def parameters(self):
        return self._params

    def half(self):
        self.half_called = True
        return self

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self

    def EnableCharacterMix(self, *_args, **_kwargs):
        self.mix_enabled = True


@pytest.fixture(autouse=True)
def _patch_torch(monkeypatch):
    class DummyCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def device_count():
            return 0

    class DummyTorch:
        cuda = DummyCuda

        @staticmethod
        def device(name):
            return name

    monkeypatch.setattr(infer_tool, "torch", DummyTorch)


@pytest.fixture
def svc_patches(monkeypatch, tmp_path):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    hps = SimpleNamespace(
        data=SimpleNamespace(
            sampling_rate=44100,
            hop_length=512,
            unit_interpolate_mode=None,
            filter_length=1024,
        ),
        train=SimpleNamespace(segment_size=4096),
        model=AttrDict(vol_embedding=False, speech_encoder="vec", inter_channels=192, hidden_channels=192, filter_channels=768, n_heads=4, n_layers=6, kernel_size=3, p_dropout=0.1, gin_channels=256, ssl_dim=256, use_spk_mix=False, use_spectral_norm=False, n_speakers=10),
        spk={"demo": 0},
    )

    monkeypatch.setattr(infer_tool.utils, "get_hparams_from_file", lambda *_: hps)
    monkeypatch.setattr(infer_tool.utils, "load_checkpoint", lambda *a, **k: (None, None, None, 0))
    monkeypatch.setattr(infer_tool.utils, "get_speech_encoder", lambda *a, **k: SimpleNamespace(encoder=lambda x: x))
    monkeypatch.setattr(infer_tool.utils, "Volume_Extractor", lambda *a, **k: "volume")
    monkeypatch.setattr(infer_tool, "SynthesizerTrn", DummySynth)
    return tmp_path


def test_svc_initializes_without_cluster(monkeypatch, svc_patches):
    svc = infer_tool.Svc(
        net_g_path="model.pth",
        config_path="config.json",
        cluster_model_path=str(svc_patches / "missing.pkl"),
        feature_retrieval=True,
    )
    assert svc.dev == "cpu"
    assert svc.feature_retrieval is False
    assert svc.spk2id == {"demo": 0}
    assert svc.unit_interpolate_mode == "left"
    assert svc.volume_extractor == "volume"
