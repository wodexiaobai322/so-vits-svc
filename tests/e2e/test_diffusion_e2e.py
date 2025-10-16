from pathlib import Path

import pytest
import soundfile as sf


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_only_diffusion_pipeline(svc_factory, diffusion_required_paths, audio_fixture_path, tmp_path_factory):
    missing = [str(path) for path in diffusion_required_paths if not path.exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")

    svc = svc_factory(only_diffusion=True)
    try:
        audio, sample_len, frame_count = svc.infer(
            speaker="demo",
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
        out_dir = Path(tmp_path_factory.mktemp("diff_only"))
        sf.write(out_dir / "only_diff.wav", audio.detach().cpu().numpy(), svc.target_sample)
    finally:
        svc.clear_empty()
        if getattr(svc, "net_g_ms", None) is not None:
            svc.unload_model()


@pytest.mark.e2e
@pytest.mark.filterwarnings('ignore:CUDA initialization:UserWarning')
@pytest.mark.filterwarnings('ignore:distutils Version classes are deprecated.*:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:Torch was not compiled with CUDA enabled.*:UserWarning')
def test_shallow_diffusion_pipeline(svc_factory, diffusion_required_paths, audio_fixture_path, tmp_path_factory):
    missing = [str(path) for path in diffusion_required_paths if not path.exists()]
    if missing:
        pytest.skip(f"缺少以下端到端所需资源：{', '.join(missing)}")

    svc = svc_factory(shallow_diffusion=True)
    try:
        audio, sample_len, frame_count = svc.infer(
            speaker="nyaru",
            tran=0,
            raw_path=str(audio_fixture_path),
            cluster_infer_ratio=0,
            auto_predict_f0=False,
            noice_scale=0.4,
            f0_filter=False,
            f0_predictor="pm",
            second_encoding=False,
            loudness_envelope_adjustment=1,
        )
        assert sample_len == audio.shape[-1]
        assert frame_count > 0
        assert float(audio.abs().max()) > 0.0
        out_dir = Path(tmp_path_factory.mktemp("diff_shallow"))
        sf.write(out_dir / "shallow_diff.wav", audio.detach().cpu().numpy(), svc.target_sample)
    finally:
        svc.clear_empty()
        if getattr(svc, "net_g_ms", None) is not None:
            svc.unload_model()
