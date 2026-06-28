import sys
import numpy as np
import pandas as pd

from typing import Optional, Union, List, Tuple, Iterable
import torch

from cantocaptions_ai.utils.audio import load_audio, SAMPLE_RATE
from cantocaptions_ai.utils.schema import TranscriptionResult, AlignedTranscriptionResult, ProgressCallback, SingleSegment
from cantocaptions_ai.utils.log_utils import get_logger

logger = get_logger(__name__)

#TODO: Better batching and inherit from Huggingface Pipeline
class SpeakerVerificationPipeline:
    def __init__(
        self,
        model_name=None,
        token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        cache_dir=None,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        if sys.platform != "linux":
            raise RuntimeError(
                "Speaker verification (--verify_speakers) requires NeMo, which is only supported on Linux. "
                "Run on a Linux system to use this feature."
            )
        import nemo.collections.asr as nemo_asr
        model_config = model_name or "nvidia/speakerverification_en_titanet_large"
        logger.info(f"Loading speaker verification model: {model_config}")

        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to(device)
        self.model.eval()

    def __call__(
        self,
        transcript: Iterable[SingleSegment],
        audio: Union[Union[str, np.ndarray], List[Union[str, np.ndarray]]],
        progress_callback: ProgressCallback = None,
    ) -> pd.DataFrame:
        """
        Perform speaker verification on multiple audios.

        Args:
            audio: Path to audio file, array, or list of audio files or arrays
        #   segment_spans: Indexes used to segment the audio data (if only one is provided)
            progress_callback: Optional callable receiving a float (0-100) with progress percentage

        Returns:
            Dataframe containing scores for each audio matching the its following audio
        """
        if isinstance(audio, str):
            audio = load_audio(audio)

        audio_data = [{
                'input_signal': torch.from_numpy(a[None, :]).to(self.device),
                'input_len': torch.tensor([torch.from_numpy(a[None, :]).shape[1]]).to(self.device)
            } for a in audio]

        all_embs = []
        for a in audio_data:
            _, embs = self.model.forward(input_signal=a["input_signal"], input_signal_length=a["input_len"])
            emb_shape = embs.shape[-1]
            embs = embs.view(-1, emb_shape)
            all_embs.append(embs.cpu().detach())

        if progress_callback is not None:
            progress_callback(100.0)

        import torch.nn.functional as F
        for i in range(0, len(all_embs)-1):
            this_speech = transcript[i]["text"]
            next_speech = transcript[i+1]["text"]
            sim = F.cosine_similarity(all_embs[i], all_embs[i+1]).item()

        diarization = F.cosine_similarity(all_embs, all_embs)

        diarize_df = pd.DataFrame(diarization.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])

        return diarize_df
