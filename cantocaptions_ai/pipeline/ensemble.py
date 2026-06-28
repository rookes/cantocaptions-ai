import os
from typing import List, Optional

from cantocaptions_ai.utils.schema import ProgressCallback, VadAudioSegment
from cantocaptions_ai.utils.model_utils import PipelineStage
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)


class FasterWhisperEnsemble(PipelineStage["List[VadAudioSegment]", "List[str]"]):
    """Second ASR model (faster-whisper) for ensemble transcription."""

    def __init__(self, model, device: str) -> None:
        self._model = model
        self._device = device

    def process(
        self,
        input: List[VadAudioSegment],
        *,
        progress_callback: ProgressCallback = None,
    ) -> List[str]:
        """Transcribe each VAD segment, returning one text string per segment."""
        if not input:
            return []

        texts = []
        n = len(input)
        for i, seg in enumerate(input):
            try:
                segments_iter, _ = self._model.transcribe(
                    seg['audio'],
                    language="yue",
                    beam_size=5,
                    vad_filter=False,
                )
                text = "".join(s.text for s in segments_iter).strip()
            except Exception as e:
                logger.warning(f"faster-whisper transcription failed on segment {i}: {e}")
                text = ""
            texts.append(text)
            if progress_callback is not None:
                progress_callback((i + 1) / n)

        return texts


def load_faster_whisper(
    model_id: str = "alvanlii/whisper-small-cantonese",
    model_subfolder: str = "cts",
    device: str = "cuda",
    device_index: int = 0,
    model_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> FasterWhisperEnsemble:
    """Load faster-whisper (CTranslate2) model and return a FasterWhisperEnsemble.

    The CTranslate2 model files are expected in the `model_subfolder` of the HuggingFace repo
    (default: 'cts' subfolder of alvanlii/whisper-small-cantonese).

    Raises ImportError if faster-whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper is required for --ensemble_model whisper: "
            "pip install 'cantocaptions_ai[ensemble]'"
        )

    import inspect as _inspect
    from huggingface_hub import snapshot_download
    import huggingface_hub.constants as _hf_const

    # Use local_dir (not cache_dir) to avoid the blob/symlink cache structure,
    # which fails on Windows without Developer Mode (WinError 1314).
    dl_root = model_dir if model_dir else _hf_const.HF_HUB_CACHE
    dl_dir = os.path.join(dl_root, model_id.replace("/", "--"))

    dl_kwargs: dict = dict(
        repo_id=model_id,
        allow_patterns=[f"{model_subfolder}/*"],
        local_dir=dl_dir,
        local_files_only=local_files_only,
    )
    # local_dir_use_symlinks exists in hf_hub 0.17–0.24; removed in 0.25+.
    if "local_dir_use_symlinks" in _inspect.signature(snapshot_download).parameters:
        dl_kwargs["local_dir_use_symlinks"] = False

    logger.info(f"Downloading faster-whisper model ({model_id}/{model_subfolder})...")
    local_path = snapshot_download(**dl_kwargs)
    ct2_path = os.path.join(local_path, model_subfolder)

    compute_type = "float16" if device == "cuda" else "int8"
    logger.info(f"Loading faster-whisper model from {ct2_path} (compute_type={compute_type})...")
    model = WhisperModel(
        ct2_path,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    return FasterWhisperEnsemble(model=model, device=device)
