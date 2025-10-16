from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_real_svc_infer_pipeline(real_svc, tmp_path, audio_fixture_path):
    audio, sample_len, frame_count = real_svc.infer(
        speaker="nyaru",
        tran=0,
        raw_path=str(audio_fixture_path),
        cluster_infer_ratio=0,
        auto_predict_f0=False,
        noice_scale=0.4,
        f0_filter=False,
        f0_predictor="pm",
    )

    assert sample_len == audio.shape[-1]
    assert frame_count > 0
    assert float(audio.abs().max()) > 0.0

    output_path = tmp_path / "svc_output.wav"
    sf.write(output_path, audio.cpu().numpy(), real_svc.target_sample)
    assert output_path.exists()


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_webui_vc_fn_with_real_model(real_svc, tmp_path, monkeypatch):
    import webUI

    webUI.model = real_svc
    Path("raw").mkdir(exist_ok=True)

    if not hasattr(real_svc, "audio_resample_transform"):
        import torchaudio

        real_svc.audio_resample_transform = torchaudio.transforms.Resample(
            real_svc.target_sample, real_svc.target_sample
        )
    if not hasattr(real_svc, "audio16k_resample_transform"):
        real_svc.audio16k_resample_transform = real_svc.audio_resample_transform

    sr = real_svc.target_sample
    duration = 3.0
    samples = np.arange(int(sr * duration))
    wave = (0.2 * np.sin(2 * np.pi * 220 * samples / sr)).astype(np.float32)
    input_path = tmp_path / "webui_long.wav"
    sf.write(input_path, wave, sr)

    def fake_slice_inference(
        self,
        raw_audio_path,
        spk,
        tran,
        slice_db,
        cluster_infer_ratio,
        auto_predict_f0,
        noice_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment,
    ):
        audio, _, _ = self.infer(
            speaker=spk,
            tran=tran,
            raw_path=raw_audio_path,
            cluster_infer_ratio=cluster_infer_ratio,
            auto_predict_f0=auto_predict_f0,
            noice_scale=noice_scale,
            f0_filter=False,
            f0_predictor="harvest",
            loudness_envelope_adjustment=loudness_envelope_adjustment,
        )
        return audio.detach().cpu().numpy()

    monkeypatch.setattr(type(real_svc), "slice_inference", fake_slice_inference, raising=False)

    message, output_path = webUI.vc_fn(
        sid="nyaru",
        input_audio=str(input_path),
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-80,
        noise_scale=0.4,
        pad_seconds=0.5,
        cl_num=0,
        lg_num=0,
        lgr_num=0.75,
        f0_predictor="pm",
        enhancer_adaptive_key=0,
        cr_threshold=0.05,
        k_step=100,
        use_spk_mix=False,
        second_encoding=False,
        loudness_envelope_adjustment=0,
    )

    assert message == "Success"
    assert output_path and output_path.endswith(".wav")
    data, sr = sf.read(output_path)
    assert sr == real_svc.target_sample
    assert np.max(np.abs(data)) > 0.0

    tmp_copy = tmp_path / Path(output_path).name
    sf.write(tmp_copy, data, sr)
    assert tmp_copy.exists()

    Path(output_path).unlink(missing_ok=True)
