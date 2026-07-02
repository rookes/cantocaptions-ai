from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the cantocaptions-ai pipeline.

    Can be constructed directly for library use or built from CLI args via
    ``PipelineConfig.from_args(vars(parsed_args))``.
    """

    # Core inference
    language: str = "yue"
    device: str = "cpu"
    device_index: int = 0
    compute_type: str = "default"
    attn_implementation: str = "sdpa"
    batch_size: int = 16
    threads: int = 0
    hf_token: Optional[str] = None

    # Model loading
    model: str = "Qwen3-ASR"
    model_dir: Optional[str] = None
    model_cache_only: bool = False

    # Output
    output_dir: str = "."
    output_format: str = "srt"
    verbose: bool = True
    print_progress: bool = True
    debug_dir: Optional[str] = None
    load_debug_dir: Optional[str] = None

    # Audio clip
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None

    # VAD
    vad_method: str = "pyannote"
    vad_onset: float = 0.500
    vad_offset: float = 0.363
    chunk_size: int = 30

    # Vocal isolation
    vocal_isolation_method: str = "mbroformer"

    # ASR options
    suppress_tokens: str = "-1"
    suppress_numerals: bool = False
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None
    condition_on_previous_text: bool = False
    fp16: bool = True

    # Ensemble & LLM correction
    ensemble_model: str = "none"
    llm_correction: bool = False
    llm_model: str = "Qwen/Qwen3-4B"
    llm_model_dir: Optional[str] = None

    # Alignment
    align_model: Optional[str] = None
    interpolate_method: str = "nearest"
    no_align: bool = False
    return_char_alignments: bool = False
    align_padding: float = 0.04
    align_release: float = 0.45
    align_merge_distance: float = 0.12

    # Subtitle formatting
    max_line_width: Optional[int] = 18
    max_line_count: Optional[int] = 2
    highlight_words: bool = False
    segment_resolution: str = "sentence"

    # Text cleaning
    no_clean_text: bool = False
    clean_rules_dir: Optional[str] = None

    # Diarization
    diarize: bool = False
    diarize_merge: bool = False
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    speaker_embeddings: bool = False

    # Speaker verification
    verify_speakers: bool = False

    # Retime
    retime: Optional[str] = None

    # Reference subtitle correction
    reference_subtitle: Optional[str] = None
    reference_correction_semantic: bool = False

    @classmethod
    def from_args(cls, args: dict) -> "PipelineConfig":
        """Build a PipelineConfig from a parsed argparse args dict.

        Unknown keys (e.g. ``audio``, ``log_level``) are silently ignored so
        this can be called on the full ``vars(parsed_args)`` dict.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in args.items() if k in known})
