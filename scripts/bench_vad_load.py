"""Break down where VAD's "load" time actually goes: imports vs. model loading vs. GPU warm-up.

Answers: VAD load consistently takes ~20-30s while the actual VAD run is only a
couple of seconds — is that model loading, GPU warm-up, or something else?
This times each phase separately to find out.

Usage:
    uv run python scripts/bench_vad_load.py
    uv run python scripts/bench_vad_load.py --device cpu
"""

import argparse
import time
from contextlib import contextmanager


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - t0:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="device to load the VAD model onto")
    parser.add_argument(
        "--model-fp", default="cantocaptions_ai/assets/pytorch_model.bin",
        help="path to the bundled Pyannote segmentation checkpoint",
    )
    args = parser.parse_args()

    with timed("import torch"):
        import torch

    with timed("import pytorch_lightning (pulled in transitively by pyannote.audio.core.Model)"):
        import pytorch_lightning  # noqa: F401

    with timed("import pyannote.audio.core.{model,inference} (lightning already loaded)"):
        # Import only from pyannote.audio.core — avoids pyannote.audio.pipelines,
        # which eagerly loads SpeakerDiarization -> speaker_verification -> NeMo.
        from pyannote.audio.core.model import Model
        from pyannote.audio.core.inference import Inference

    with timed("Model.from_pretrained (CPU)"):
        vad_model = Model.from_pretrained(args.model_fp, token=None)

    with timed(f"Inference(...) construction (incl. .to({args.device}))"):
        inference = Inference(
            vad_model,
            device=torch.device(args.device),
            pre_aggregation_hook=lambda scores: scores,
        )

    import numpy as np
    audio = torch.from_numpy(np.random.randn(1, 16000 * 30).astype("float32"))

    with timed("first inference call (incl. any CUDA kernel warm-up)"):
        inference({"waveform": audio, "sample_rate": 16000})
        if args.device == "cuda":
            torch.cuda.synchronize()

    with timed("second inference call (steady state)"):
        inference({"waveform": audio, "sample_rate": 16000})
        if args.device == "cuda":
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
