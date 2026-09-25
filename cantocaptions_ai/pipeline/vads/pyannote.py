import os

import numpy as np
import torch

# Binarize is re-exported: it lived here before the other score-curve backends existed, and
# callers outside this package (tests, cantocaptions-dataset) still import it from here.
from cantocaptions_ai.pipeline.vads.curve import Binarize as Binarize, CurveVad
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


def load_vad_model(device, token=None, model_fp=None):
    """Load the pyannote segmentation model as a frame-scoring Inference.

    Note this only produces the score curve; the onset/offset thresholds and all smoothing
    are applied later, in Binarize (via merge_chunks) — not here.
    """
    # Import only from pyannote.audio.core — avoids pyannote.audio.pipelines.__init__,
    # which eagerly loads SpeakerDiarization → speaker_verification → NeMo.
    from pyannote.audio.core.model import Model
    from pyannote.audio.core.inference import Inference

    model_dir = torch.hub._get_torch_home()

    # __file__ is cantocaptions_ai/pipeline/vads/pyannote.py
    # assets/ is at cantocaptions_ai/assets/ (3 levels up)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    os.makedirs(model_dir, exist_ok=True)
    if model_fp is None:
        model_fp = os.path.join(main_dir, "assets", "pytorch_model.bin")
        model_fp = os.path.abspath(model_fp)
    else:
        model_fp = os.path.abspath(model_fp)

    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"Model file not found at {model_fp}")

    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    vad_model = Model.from_pretrained(model_fp, token=token)
    return Inference(
        vad_model,
        device=torch.device(device),
        pre_aggregation_hook=lambda scores: np.max(scores, axis=-1, keepdims=True),
    )


class Pyannote(CurveVad):

    def __init__(self, device, token=None, model_fp=None, **kwargs):
        logger.info("Performing voice activity detection using Pyannote...")
        super().__init__(kwargs['vad_onset'])
        self.vad_pipeline = load_vad_model(device, token=token, model_fp=model_fp)

    def __call__(self, audio, **kwargs):
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio):
        return torch.from_numpy(audio).unsqueeze(0)
