import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("gradio")
pytest.importorskip("gradio_client")

import gradio as gr  # noqa: E402
from gradio_client import Client  # noqa: E402

warnings.simplefilter("ignore", DeprecationWarning)

os.environ.setdefault("SOVITS_WEBUI_HEADLESS", "1")

import webUI  # noqa: E402


@pytest.mark.e2e
@pytest.mark.filterwarnings(r"ignore:websockets\.legacy is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings(r"ignore:websockets\.server.WebSocketServerProtocol is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated:DeprecationWarning")
def test_vc_fn_end_to_end(monkeypatch, tmp_path):
    class DummySvc:
        def __init__(self):
            self.spk2id = {"demo": 0}
            self.dev = "cpu"
            self.shallow_diffusion = False
            self.only_diffusion = False
            self.feature_retrieval = False
            self.cluster_model = None
            self.target_sample = 16000
            self.slice_calls = []
            self.cleared = False

        def slice_inference(
            self,
            audio_path,
            spk,
            vc_transform,
            slice_db,
            cluster_infer_ratio,
            auto_f0,
            noise_scale,
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
            self.slice_calls.append(
                {
                    "audio_path": audio_path,
                    "spk": spk,
                    "vc_transform": vc_transform,
                    "slice_db": slice_db,
                    "cluster_ratio": cluster_infer_ratio,
                    "noise_scale": noise_scale,
                }
            )
            t = np.linspace(0, 2 * math.pi, self.target_sample, endpoint=False)
            return (0.2 * np.sin(440 * t)).astype(np.float32)

        def clear_empty(self):
            self.cleared = True

        def unload_model(self):
            pass

    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw").mkdir()
    dummy_model = DummySvc()
    monkeypatch.setattr(webUI, "model", dummy_model, raising=False)
    class _DummyMatplotlibManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "gradio.utils.MatplotlibBackendMananger",
        lambda: _DummyMatplotlibManager(),
    )

    class _DummyPandas:
        @staticmethod
        def DataFrame(data):
            return data

    monkeypatch.setattr("gradio.queueing.pd", _DummyPandas(), raising=False)

    def wrapped(input_audio):
        return webUI.vc_fn(
            sid="demo",
            input_audio=input_audio,
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

    demo = gr.Interface(
        fn=wrapped,
        inputs=gr.Textbox(label="audio path"),
        outputs=[gr.Textbox(label="message"), gr.Textbox(label="path")],
        flagging_mode="never",
    )

    demo.launch(server_name="127.0.0.1", server_port=None, prevent_thread_lock=True, share=False)
    try:
        port = demo.server_port
        assert port is not None
        client = Client(f"http://127.0.0.1:{port}/")

        audio_path = Path("input.wav")
        t = np.linspace(0, 1, 16000, endpoint=False)
        sf.write(audio_path, 0.1 * np.sin(2 * np.pi * 220 * t), 16000)

        message, result_path = client.predict(
            str(audio_path),
            api_name="/predict",
        )

        assert message == "Success"
        assert result_path.endswith(".wav")
        output_file = Path(result_path)
        assert output_file.exists()
        data, sr = sf.read(output_file)
        assert sr == dummy_model.target_sample
        assert data.ndim == 1
        assert dummy_model.cleared
        assert dummy_model.slice_calls
    finally:
        demo.close()
