import pytest


def test_vc_fn_requires_audio(webui_module):
    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio=None,
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0,
        slice_db=-40,
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
    assert msg == "You need to upload an audio"
    assert output is None


def test_vc_fn_cluster_ratio_requires_model(monkeypatch, webui_module):
    webui_module.model = None
    msg, output = webui_module.vc_fn(
        sid="demo",
        input_audio="dummy.wav",
        output_format="wav",
        vc_transform=0,
        auto_f0=False,
        cluster_ratio=0.5,
        slice_db=-40,
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
    assert "You need to upload an model" in msg or "cluster model" in msg
    assert output is None


def test_vc_fn_feature_retrieval_missing_index(monkeypatch, webui_module, tmp_path):
    class DummyModel:
        feature_retrieval = True
        cluster_model = {}
        spk2id = {"demo": 0}
        target_sample = 16000
        shallow_diffusion = False
        only_diffusion = False

        def slice_inference(self, *args, **kwargs):
            raise AssertionError("should not be called when index missing")

    webui_module.model = DummyModel()

    with pytest.raises(webui_module.gr.Error):
        webui_module.vc_fn(
            sid="demo",
            input_audio=str(tmp_path / "clip.wav"),
            output_format="wav",
            vc_transform=0,
            auto_f0=False,
            cluster_ratio=0.2,
            slice_db=-40,
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


def test_vc_fn_invalid_audio_path(monkeypatch, webui_module):
    class DummyModel:
        feature_retrieval = False
        cluster_model = None

    webui_module.model = DummyModel()
    with pytest.raises(webui_module.gr.Error):
        webui_module.vc_fn(
            sid="demo",
            input_audio="missing.wav",
            output_format="wav",
            vc_transform=0,
            auto_f0=False,
            cluster_ratio=0,
            slice_db=-40,
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
