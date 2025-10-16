import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _skip_if_missing(paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")


class FlaskSvcAdapter:
    def __init__(self, svc):
        self._svc = svc
        self.target_sample = svc.target_sample
        self._id2spk = {int(v): k for k, v in svc.spk2id.items()}

    def infer(self, speaker_id, pitch, wav_buffer, **kwargs):
        speaker = self._id2spk.get(int(float(speaker_id)), next(iter(self._id2spk.values())))
        wav_buffer.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_buffer.read())
            tmp_path = tmp.name
        try:
            audio, sample_len, _ = self._svc.infer(
                speaker=speaker,
                tran=int(float(pitch)),
                raw_path=tmp_path,
                cluster_infer_ratio=kwargs.get("cluster_infer_ratio", 0),
                auto_predict_f0=kwargs.get("auto_predict_f0", False),
                noice_scale=kwargs.get("noice_scale", 0.4),
                f0_filter=kwargs.get("f0_filter", False),
                f0_predictor=kwargs.get("f0_predictor", "pm"),
            )
            return audio, sample_len
        finally:
            os.remove(tmp_path)

    def clear_empty(self):
        self._svc.clear_empty()

    def unload_model(self):
        self._svc.unload_model()


class FlaskSongAdapter:
    def __init__(self, svc):
        self._svc = svc
        self.target_sample = svc.target_sample

    def infer(self, spk, tran, raw_path):
        audio, sample_len, _ = self._svc.infer(
            speaker=spk,
            tran=int(float(tran)),
            raw_path=raw_path,
            cluster_infer_ratio=0,
            auto_predict_f0=False,
            noice_scale=0.4,
            f0_filter=False,
            f0_predictor="pm",
        )
        return audio, sample_len

    def clear_empty(self):
        self._svc.clear_empty()


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_flask_voice_change_endpoint(real_svc, base_required_paths, audio_fixture_path):
    _skip_if_missing(base_required_paths)
    import flask_api

    flask_api.raw_infer = True
    flask_api.svc_model = FlaskSvcAdapter(real_svc)

    buffer = io.BytesIO()
    data, sr = sf.read(audio_fixture_path)
    sf.write(buffer, data, sr, format="wav")
    buffer.seek(0)

    client = flask_api.app.test_client()
    response = client.post(
        "/voiceChangeModel",
        data={
            "fPitchChange": "0",
            "sampleRate": str(real_svc.target_sample),
            "sSpeakId": "0",
            "sample": (buffer, "input.wav"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"] in {"audio/wav", "audio/x-wav"}
    assert len(response.data) > 1000
    output_buffer = io.BytesIO(response.data)
    output_buffer.seek(0)
    out_audio, out_sr = sf.read(output_buffer)
    assert out_sr == real_svc.target_sample
    assert np.max(np.abs(out_audio)) > 0.0


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_flask_wav2wav_endpoint(tmp_path_factory, base_required_paths, audio_fixture_path, svc_factory):
    _skip_if_missing(base_required_paths)
    import flask_api_full_song

    svc = svc_factory()
    flask_api_full_song.svc_model = FlaskSongAdapter(svc)

    client = flask_api_full_song.app.test_client()
    song_path = tmp_path_factory.mktemp("flask") / "song.wav"
    data, sr = sf.read(audio_fixture_path)
    sf.write(song_path, data, sr)

    response = client.post(
        "/wav2wav",
        data={
            "audio_path": str(song_path),
            "tran": "0",
            "spk": "nyaru",
            "wav_format": "wav",
        },
    )

    try:
        assert response.status_code == 200
        assert response.headers["Content-Type"] in {"audio/wav", "audio/x-wav"}
        buffer = io.BytesIO(response.data)
        buffer.seek(0)
        out_audio, out_sr = sf.read(buffer)
        assert out_sr == svc.target_sample
        assert np.max(np.abs(out_audio)) > 0.0
    finally:
        svc.clear_empty()
        svc.unload_model()
