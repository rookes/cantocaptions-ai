from dataclasses import dataclass, field, fields, MISSING
from typing import Any, Dict, Optional


def _detect_default_device() -> str:
    """Best available torch device: cuda > mps > cpu.

    A function (not a static default) since the right value depends on the
    machine PipelineConfig is constructed on.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PipelineConfig:
    """Configuration for the cantocaptions-ai pipeline.

    Can be constructed directly for library use or built from CLI args via
    ``PipelineConfig.from_args(vars(parsed_args))``. Field defaults here are
    the single source of truth for the pipeline's baseline behavior: the CLI
    (``cantocaptions_ai/__main__.py``) carries no defaults of its own — it
    reads them from ``PipelineConfig.defaults()`` for ``--help`` display and
    for the config-file/preset layering in ``pipeline/cli_config.py``.
    """

    # Core inference
    language: str = "yue"
    device: str = field(default_factory=_detect_default_device)
    device_index: int = 0
    asr_compute_type: str = "default"
    attn_implementation: str = "sdpa"
    batch_size: int = 15
    threads: int = 0
    hf_token: Optional[str] = None
    compile: bool = False

    # Model loading
    model: str = "Qwen3-ASR"
    model_dir: Optional[str] = None
    model_cache_only: bool = False

    # Output
    output_dir: str = "."
    output_format: str = "srt"
    verbose: bool = True
    print_progress: bool = True
    vram_checks: bool = True
    vram_headroom_mb: int = 512
    debug_dir: Optional[str] = None
    load_debug_dir: Optional[str] = None

    # Audio clip
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None

    # VAD
    vad_method: str = "pyannote"
    vad_onset: float = 0.450
    vad_offset: float = 0.300
    vad_pad_onset: float = 0.20
    vad_pad_offset: float = 0.20
    vad_min_duration_off: float = 0.25
    chunk_size: int = 30

    # Vocal isolation
    # Off by default: the Mel-Band RoFormer stage is a heavy add for a small gain on clean
    # speech. Opt in with --vocal_isolation_method mbroformer for noisy/music-heavy audio.
    vocal_isolation_method: str = "none"
    vocal_isolation_batch_size: int = 4
    vocal_isolation_compute_type: str = "float32"

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
    align_release: float = 0.64
    align_merge_distance: float = 0.08
    min_cue_duration: float = 0.5
    merge_gap: float = 0.25
    align_batch_size: int = 4
    align_compute_type: str = "float16"
    # Substitute an in-vocabulary character for one the align model has no token for, so the
    # trellis can see it at all. "homophone" (same Jyutping reading) is the default because a
    # dropped character contributes no evidence whatsoever; "near" also accepts the same
    # syllable on another tone, "variant" folds Simplified forms only, "off" disables it.
    # See pipeline/align_vocab.py.
    align_char_substitution: str = "homophone"
    # TOML file of hand-curated substitutions that beat every automatic tier.
    align_substitutions: Optional[str] = None

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
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    # "segment" diarizes each VAD segment independently (speaker labels namespaced per
    # segment, only comparable within one); "file" diarizes the whole file in one pass.
    diarize_scope: str = "segment"
    # Chunks per diarization forward pass. NOT the checkpoint's own value: community-1's
    # config.yaml asks for 32, whose in-call peak measured 10.7 GB on a 28s segment. That
    # does not fit a 10 GB card, and Windows answers an oversubscribed allocation by paging
    # into host RAM rather than raising -- so it does not fail, it runs ~50x slower. Measured
    # on an RTX 3080, one 28s segment: batch 32 -> 10.7 GB / 60s, 16 -> 10.0 GB / 45s,
    # 8 -> 9.4 GB / 585s, 4 -> 251 MB / 1.09s. The jump between 4 and 8 is a kernel workspace
    # threshold, not smooth scaling, so 4 is the safe side of a cliff rather than a tuning
    # knob. Raise it only if you have measured the peak on your own card.
    diarize_batch_size: Optional[int] = 4
    speaker_embeddings: bool = False
    # Share of a subsegment's diarized time the leading speaker must hold before the
    # subsegment is attributed at all. Below it the subsegment stays unlabeled, which cue
    # assembly reads as "no objection to merging" -- see pipeline/speaker_assign.py.
    speaker_confidence: float = 0.7
    # Share the runner-up must hold for a subsegment to be flagged as multi-speaker.
    speaker_conflict_share: float = 0.25
    flag_speaker_conflicts: bool = False
    speaker_labels: bool = False

    # Retime
    retime: Optional[str] = None

    # Realign: put an untimed transcript on the audio timeline. See pipeline/realign.py.
    realign: Optional[str] = None
    # 'acoustic' places lines with a sliding free-end Viterbi and no ASR; 'asr' runs the
    # normal ASR stage and matches the two character streams, which is slower but degrades
    # gracefully when the transcript and the recording disagree.
    realign_anchor: str = "acoustic"
    # Seconds of audio each placement window sees. Larger windows give the search more
    # context to resynchronise in; the cost is quadratic only in the trellis, which is cheap.
    realign_window: float = 120.0
    # Lines ending within this far of the window's far edge are held for the next window,
    # since a line straddling the edge has only been seen in part.
    realign_commit_margin: float = 10.0
    # Mean CTC path score below which a placed line is reported as weakly supported. This is
    # a diagnostic only -- nothing is dropped or retimed on the strength of it.
    realign_min_score: float = 0.35

    # How a multichannel source is reduced to mono. "center" takes the front-center
    # channel alone, which on a film soundtrack is largely the dialogue stem; it falls
    # back to a full downmix for any layout without one. See utils/audio.py.
    audio_downmix: str = "mix"

    # Reference subtitle correction
    reference_subtitle: Optional[str] = None
    reference_correction_semantic: bool = False
    # Constant offset applied to every reference cue before use, for a reference sourced
    # from a different release or OCR'd with a systematic lag. Affects both consumers.
    reference_offset: float = 0.0

    # Reference subtitle as ASR context (experimental). Routes the same
    # --reference_subtitle file into Qwen3-ASR's context-biasing system prompt and,
    # separately, into the VAD timeline. See pipeline/reference_context.py.
    asr_context: bool = False
    asr_context_template: str = "labelled"
    # 'all' prompts every segment with the cues covering it; 'expanded' prompts only over
    # audio the reference recovered that VAD missed, leaving VAD's own detections bare.
    asr_context_scope: str = "all"
    asr_context_neighbours: int = 0
    # Context is a prefill cost paid per segment per batch; this caps it.
    asr_context_max_chars: int = 400
    asr_context_vad_expand: bool = True
    # Seconds added either side of every reference cue before it is unioned into the VAD
    # timeline. Every cue is included -- there is no confidence gate -- so this is the
    # only knob controlling how much audio the reference contributes.
    asr_context_padding: float = 0.5

    @classmethod
    def from_args(cls, args: dict) -> "PipelineConfig":
        """Build a PipelineConfig from a parsed argparse args dict.

        Unknown keys (e.g. ``audio``, ``log_level``) are silently ignored so
        this can be called on the full ``vars(parsed_args)`` dict.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in args.items() if k in known})

    @classmethod
    def defaults(cls) -> Dict[str, Any]:
        """Every field's baseline default, resolving default_factory fields
        (currently only ``device``).

        The one place ``--help`` text, config/default.cfg auto-generation,
        and the base layer of the CLI's config-file/preset merge all read
        their baseline values from.
        """
        out: Dict[str, Any] = {}
        for f in fields(cls):
            if f.default is not MISSING:
                out[f.name] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                out[f.name] = f.default_factory()
            else:
                raise TypeError(f"PipelineConfig.{f.name} has no default")
        return out
