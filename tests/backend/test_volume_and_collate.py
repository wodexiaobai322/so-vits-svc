import numpy as np
import torch

import utils
import data_utils


def test_volume_extractor_accepts_numpy_audio():
    extractor = utils.Volume_Extractor(hop_size=4)
    audio = np.ones((1, 16), dtype=np.float32)

    volume = extractor.extract(audio)

    assert isinstance(volume, torch.Tensor)
    assert volume.shape[0] == 4
    assert torch.all(volume >= 0)


def test_volume_extractor_respects_tensor_input():
    extractor = utils.Volume_Extractor(hop_size=8)
    audio = torch.linspace(-1, 1, steps=32).view(1, -1)

    volume = extractor.extract(audio)

    assert volume.shape[0] == 4
    assert torch.isfinite(volume).all()


def _make_sample(channel_count, length, fill_value, speaker_id, with_volume=True):
    c = torch.full((channel_count, length), fill_value, dtype=torch.float32)
    f0 = torch.linspace(0.0, 1.0, steps=length)
    spec = torch.full((channel_count, length), fill_value, dtype=torch.float32)
    wav = torch.full((1, length * 2), fill_value, dtype=torch.float32)
    spk = torch.tensor([speaker_id], dtype=torch.long)
    uv = torch.zeros(length, dtype=torch.float32)
    volume = torch.full((length,), fill_value, dtype=torch.float32) if with_volume else None
    return c, f0, spec, wav, spk, uv, volume


def test_text_audio_collate_pads_and_orders_by_length():
    collate = data_utils.TextAudioCollate()
    long_sample = _make_sample(channel_count=2, length=3, fill_value=1.0, speaker_id=7)
    short_sample = _make_sample(channel_count=2, length=2, fill_value=2.0, speaker_id=3)

    c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded = collate(
        [short_sample, long_sample]
    )

    assert c_padded.shape == (2, 2, 3)
    assert torch.equal(lengths, torch.tensor([3, 2]))
    assert spkids[0, 0].item() == 7
    assert torch.allclose(volume_padded[0, :3], torch.ones(3))
    assert torch.allclose(volume_padded[1, :2], torch.full((2,), 2.0))
    assert wav_padded.shape[-1] == 6
    assert torch.equal(f0_padded[0, :3], torch.linspace(0.0, 1.0, steps=3))


def test_text_audio_collate_returns_none_when_volume_missing():
    collate = data_utils.TextAudioCollate()
    with_volume = _make_sample(channel_count=1, length=3, fill_value=0.5, speaker_id=1, with_volume=True)
    without_volume = _make_sample(channel_count=1, length=2, fill_value=1.5, speaker_id=2, with_volume=False)

    result = collate([with_volume, without_volume])

    assert result[-1] is None
