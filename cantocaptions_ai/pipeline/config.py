from dataclasses import dataclass, field, fields, MISSING
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple


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


# Kept here rather than in pipeline/vads so the CLI can offer it as choices without
# importing torch at startup.
VAD_METHODS = ("pyannote", "silero")


@dataclass
class PipelineConfig:
    """Configuration for the cantocaptions-ai pipeline.

    Can be constructed directly for library use or built from CLI args via
    ``PipelineConfig.from_args(vars(parsed_args))``. Field defaults here are
    the single source of truth for the pipeline's baseline behavior: the CLI
    (``cantocaptions_ai/__main__.py``) carries no defaults of its own — it
    reads them from ``PipelineConfig.defaults()`` for ``--help`` display and
    for the config-file/preset layering in ``pipeline/cli_config.py``.

    These are kept in step with the shipped ``config/default.cfg``, which sits
    one layer above them. They are not redundant: the cfg file is what a CLI
    user edits, while these are what a library caller and ``--help`` see, so a
    divergence makes ``--help`` state a default no CLI run actually uses. If
    you change one, change the other; ``tests/test_cli_config.py`` pins it.
    """

    # Core inference
    language: str = "yue"
    device: str = field(default_factory=_detect_default_device)
    device_index: int = 0
    asr_compute_type: str = "default"
    attn_implementation: str = "sdpa"
    batch_size: int = 8
    threads: int = 0
    hf_token: Optional[str] = None
    compile: bool = False

    # Model loading
    # The ASR model; None uses the language pack's own (cantocaptions-cantonese-ASR for yue).
    model: Optional[str] = None
    model_dir: Optional[str] = None
    model_cache_only: bool = False

    # Output
    output_dir: str = "output"
    output_format: str = "srt"
    verbose: bool = True
    print_progress: bool = True
    vram_checks: bool = False
    vram_headroom_mb: int = 0
    debug_dir: Optional[str] = None
    load_debug_dir: Optional[str] = None

    # Audio clip
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None

    # VAD
    vad_method: str = "pyannote"
    vad_onset: float = 0.15
    vad_offset: float = 0.15
    vad_pad_onset: float = 0.25
    vad_pad_offset: float = 0.20
    vad_min_duration_off: float = 0.25
    chunk_size: int = 28

    # Vocal isolation
    # Off by default: the Mel-Band RoFormer stage is a heavy add for a small gain on clean
    # speech. Opt in with --vocal_isolation_method mbroformer for noisy/music-heavy audio.
    vocal_isolation_method: str = "none"
    vocal_isolation_batch_size: int = 4
    vocal_isolation_compute_type: str = "float32"
    # How the isolation model consumes a segment. Must match whatever the ASR
    # model was trained on: a model trained on whole-mode isolated audio meets
    # different artifacts under chunked mode. (Direct load_vocal_isolation callers
    # that pass segment_mode=None instead defer to the bundled model yaml.)
    vocal_isolation_segment_mode: str = "chunked"

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
    # Longest a cue may run, in seconds: joins that would pass it are refused, and a cue
    # alignment produced longer is cut at its longest internal pause
    # (segmentation._split_long_cues). 0 turns the cap off.
    max_cue_duration: float = 4.0
    merge_gap: float = 0.25
    align_batch_size: int = 2
    align_compute_type: str = "float16"
    # Substitute an in-vocabulary character for one the align model has no token for, so the
    # trellis can see it at all. "homophone" (same Jyutping reading) because a dropped
    # character contributes no evidence whatsoever; "near" also accepts the same syllable on
    # another tone, "variant" folds Simplified forms only, "off" disables it. None (the
    # default) takes the align model's profile: homophone for the Cantonese model, off for
    # any other, since the homophone tiers read Cantonese pronunciations.
    # See pipeline/align_vocab.py and align_profiles.AlignProfile.char_substitution.
    align_char_substitution: Optional[str] = None
    # TOML file of hand-curated substitutions that beat every automatic tier.
    align_substitutions: Optional[str] = None
    # Break a cue in two wherever alignment left a silence of at least this many seconds
    # between two of its own adjacent characters -- a cue that holds two utterances has a
    # start or an end that was never spoken. None defers to the align model's own profile
    # (AlignProfile.split_gap, itself None for every shipped model, so: never); 0 forces it
    # off whatever the profile says. See pipeline/align_checks.py:split_gapped_cues.
    align_split_gap: Optional[float] = None

    # Subtitle formatting
    # Characters per line before text is broken onto the next. None: the language's own width
    # (its script's line_width: 18 for Chinese, 42 for space-separated scripts), filled in by
    # validate_config (see LANGUAGE_DEFAULTED). 0: never break a line.
    max_line_width: Optional[int] = None
    max_line_count: Optional[int] = 2

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

    # Realign: put a transcript on the audio timeline. See pipeline/realign.py.
    realign: Optional[str] = None
    # What to do with the input's own timings, if it has any.
    #   transcript -- discard them and place every line from scratch
    #   sync       -- fit a transform from the audio and map every cue through it, which
    #                 preserves the subtitle's proportions exactly
    #   adjust     -- fit the same transform, then re-time each cue from the audio inside
    #                 realign_adjust_tolerance of where the transform put it
    #   auto       -- transcript for a bare transcript, sync for a subtitle with timings
    # This replaces the old --retime, which could only carry a rolling offset and so could
    # not express a rate change at all. See pipeline/timefit.py.
    realign_mode: str = "auto"
    # Bound on |scale - 1| the fitted transform may use. Beyond it the fit is refused rather
    # than clamped, since a clamped scale is a number the fit does not believe.
    realign_max_scale: float = 0.25
    # What happens to cues the fit says this recording has no audio for.
    #   drop -- leave them out of the output entirely, and report them
    #   keep -- collapse them onto the cut point, flagged, so nothing the user wrote is lost
    realign_cut_policy: str = "drop"
    # Mode adjust only: how far forced alignment may move a cue from its transform position
    # before the transform wins. Widened automatically when the transform fits only loosely.
    realign_adjust_tolerance: float = 2.0
    # Fold the input's punctuation into the forms the aligner can use (halfwidth marks beside
    # Chinese text to fullwidth, any ellipsis to U+2026). On by default because most such
    # marks are typing slips and because the aligner spends a real pause on the result. Turn
    # it off to have the text reach the writer exactly as written -- at the cost of alignment,
    # since a halfwidth mark is in neither the align vocabulary nor split_chars and is simply
    # dropped. See realign.normalize_transcript_text.
    realign_normalize: bool = True
    # Mode sync only: roughly how many anchors per minute of the file get to set a cue's
    # start directly, once thinned by residual against the fitted transform (see
    # timefit.prune_to_density). Every other cue is interpolated between whichever of those
    # survive nearby. Keeping every anchor exposes every cue to that anchor's own acoustic
    # search individually -- a confidently-wrong anchor (a repeated phrase confusing CTC, a
    # stylised reading) scores exactly as well as a correct one, so a confidence floor cannot
    # separate them; the fitted transform's own residual can.
    realign_sync_anchor_density: float = 3.0
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

    # 0-based audio stream to transcribe. None picks one from the stream tags by language
    # (utils/audio.select_track); set it when a release's tags are missing or wrong.
    audio_track: Optional[int] = None

    # Bring each file's speech to a fixed level with one linear gain before any
    # stage sees it. On by default because the TRAINING corpus is cut this way:
    # a model trained on levelled audio and then shown unlevelled audio meets a
    # different distribution than it learned. See utils/audio.normalize_gain.
    audio_normalize: bool = True

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

    @staticmethod
    def section_of(name: str) -> str:
        """The section (a ``CONFIG_SECTIONS`` key) that setting *name* belongs to."""
        try:
            return _SECTION_OF[name]
        except KeyError:
            raise KeyError(f"no pipeline setting named {name!r}") from None

    def section(self, name: str) -> Mapping[str, Any]:
        """A read-only view of one section's settings, by their flat names."""
        return MappingProxyType({key: getattr(self, key) for key in CONFIG_SECTIONS[name]})

    @staticmethod
    def flatten(nested: Mapping[str, Any]) -> Dict[str, Any]:
        """Flat settings from a mapping that may group them by section.

        ``{"vad": {"vad_onset": 0.15}, "language": "yue"}`` gives
        ``{"vad_onset": 0.15, "language": "yue"}``: a section's table holds that section's
        settings under their usual flat names, and flat keys may sit beside the tables. A key
        in the wrong section, one given twice, or one that is no setting at all raises
        ValueError -- the same rules as a sectioned .cfg file.
        """
        out: Dict[str, Any] = {}

        def put(key: str, value: Any, where: str) -> None:
            if key not in _SECTION_OF:
                raise ValueError(f"{where}: unknown pipeline setting {key!r}")
            if key in out:
                raise ValueError(f"{where}: {key!r} is set twice")
            out[key] = value

        for key, value in nested.items():
            if key in CONFIG_SECTIONS and isinstance(value, Mapping):
                for inner, inner_value in value.items():
                    home = _SECTION_OF.get(inner)
                    if home is not None and home != key:
                        raise ValueError(
                            f"[{key}]: {inner!r} belongs in [{home}], not [{key}]")
                    put(inner, inner_value, f"[{key}]")
            else:
                put(key, value, "top level")
        return out


# --- Sections --------------------------------------------------------------------------
#
# Every setting belongs to exactly one section: the stage or concern it configures. The flat
# field names above stay the only names a setting has -- in flags, .cfg keys, the worker's
# TOML, checkpoint fingerprints -- and a section only groups them: in --help (each section
# is one argument group, titled as below), in .cfg files (a [vad] block may hold the VAD
# settings), and for callers that want one stage's settings (PipelineConfig.section).
# tests/test_cli_config.py pins this table to the fields and to the CLI's argument groups.

SECTION_TITLES: Mapping[str, str] = MappingProxyType({
    "model": "model",
    "inference": "inference",
    "output": "output",
    "audio": "audio",
    "vad": "vad",
    "vocal_isolation": "vocal isolation",
    "ensemble": "ensemble & LLM correction",
    "asr_context": "asr context (experimental)",
    "alignment": "alignment",
    "cues": "cue timing",
    "subtitles": "subtitle formatting",
    "cleaning": "text cleaning",
    "diarization": "diarization",
    "realign": "existing transcript",
})

CONFIG_SECTIONS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "model": ("language", "model", "model_dir", "model_cache_only"),
    "inference": (
        "device", "device_index", "batch_size", "asr_compute_type", "attn_implementation",
        "threads", "hf_token", "compile",
    ),
    "output": (
        "output_dir", "output_format", "verbose", "print_progress", "vram_checks",
        "vram_headroom_mb", "debug_dir", "load_debug_dir",
    ),
    "audio": ("audio_start", "audio_end", "audio_downmix", "audio_track", "audio_normalize"),
    "vad": (
        "vad_method", "vad_onset", "vad_offset", "vad_pad_onset", "vad_pad_offset",
        "vad_min_duration_off", "chunk_size",
    ),
    "vocal_isolation": (
        "vocal_isolation_method", "vocal_isolation_batch_size", "vocal_isolation_compute_type",
        "vocal_isolation_segment_mode",
    ),
    "ensemble": (
        "ensemble_model", "llm_correction", "llm_model", "llm_model_dir", "reference_subtitle",
        "reference_correction_semantic", "reference_offset",
    ),
    "asr_context": (
        "asr_context", "asr_context_template", "asr_context_scope", "asr_context_neighbours",
        "asr_context_max_chars", "asr_context_vad_expand", "asr_context_padding",
    ),
    "alignment": (
        "align_model", "interpolate_method", "no_align", "return_char_alignments",
        "align_batch_size", "align_compute_type", "align_char_substitution",
        "align_substitutions", "align_split_gap",
    ),
    "cues": (
        "align_padding", "align_release", "align_merge_distance", "min_cue_duration",
        "max_cue_duration", "merge_gap",
    ),
    "subtitles": ("max_line_width", "max_line_count"),
    "cleaning": ("no_clean_text", "clean_rules_dir"),
    "diarization": (
        "diarize", "min_speakers", "max_speakers", "diarize_model", "diarize_scope",
        "diarize_batch_size", "speaker_embeddings", "speaker_confidence",
        "speaker_conflict_share", "flag_speaker_conflicts", "speaker_labels",
    ),
    "realign": (
        "realign", "realign_mode", "realign_max_scale", "realign_cut_policy",
        "realign_adjust_tolerance", "realign_normalize", "realign_sync_anchor_density",
        "realign_anchor", "realign_window", "realign_commit_margin", "realign_min_score",
    ),
})

# Settings whose unset (None) value means "whatever the language does": validate_config fills
# each in from the run profile (the language pack resolved for the ASR model), so a
# Cantonese value never becomes another language's default by accident.
LANGUAGE_DEFAULTED: Mapping[str, Any] = MappingProxyType({
    "max_line_width": lambda profile: profile.script.line_width,
})

_SECTION_OF: Dict[str, str] = {
    key: section for section, keys in CONFIG_SECTIONS.items() for key in keys
}
