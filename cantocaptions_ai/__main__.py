import argparse
import functools
import importlib.metadata
import platform

from cantocaptions_ai.pipeline.cli_config import ConfigAwareHelpFormatter, resolve_pipeline_args
from cantocaptions_ai.pipeline.config import PipelineConfig
from cantocaptions_ai.utils.output import (LANGUAGES, TO_LANGUAGE_CODE,
                            optional_int, str2bool)
from cantocaptions_ai.utils.log_utils import setup_logging, get_logger

logger = get_logger(__name__)

_MEDIA_EXTENSIONS = {
    '.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts', '.m2ts',
    '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.opus',
}


def discover_media_files(directory: str, recursive: bool = False) -> list:
    """Return sorted list of media file paths found in *directory*.

    Searches for files whose extension is in _MEDIA_EXTENSIONS. Raises
    ValueError if *directory* does not exist or contains no eligible files.
    """
    from pathlib import Path
    base = Path(directory)
    if not base.is_dir():
        raise ValueError(f"--input_dir '{directory}' is not a directory or does not exist")
    pattern = "**/*" if recursive else "*"
    files = sorted(
        str(p) for p in base.glob(pattern)
        if p.is_file() and p.suffix.lower() in _MEDIA_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No eligible media files found in '{directory}'")
    return files


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI's argparse parser.

    Every flag's ``default`` is ``argparse.SUPPRESS`` — PipelineConfig
    (pipeline/config.py) is the single source of truth for baseline defaults,
    and cli_config.resolve_pipeline_args() layers config-file/preset/explicit
    values on top of it. SUPPRESS also makes "was this flag explicitly typed"
    detectable at all: without it, argparse would always populate every field
    with its own default, and there'd be no way to tell a config file's value
    from an explicit override.
    """
    # fmt: off
    formatter_class = functools.partial(ConfigAwareHelpFormatter, defaults=PipelineConfig.defaults())
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    parser.add_argument("audio", nargs="*", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--language", type=str, default=argparse.SUPPRESS, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--retime", type=str, default=argparse.SUPPRESS, metavar="SUBTITLE_FILE", help="subtitle file to retime against the audio (SRT, VTT, etc.). Skips ASR; updates timings only, text is preserved.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('cantocaptions-ai')}", help="Show cantocaptions-ai version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})", help="Show python version information and exit")

    config_grp = parser.add_argument_group("config file")
    config_grp.add_argument("--cfg", type=str, default=argparse.SUPPRESS, metavar="NAME", help="load config/NAME.cfg instead of the auto-created config/default.cfg (see config/cpu.cfg for a commented example); its values act as a layer beneath stage-preset and explicit CLI flags")

    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--model", default=argparse.SUPPRESS, choices=["Qwen3-ASR", "Qwen3-ASR-0.6B"], help="name of the model to use" "Qwen3-ASR")
    model_grp.add_argument("--model_cache_only", type=str2bool, default=argparse.SUPPRESS, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    model_grp.add_argument("--model_dir", type=str, default=argparse.SUPPRESS, help="the path to save/load model files; if unset, defers to huggingface_hub's own cache resolution (~/.cache/huggingface/hub, or $HF_HOME/$XDG_CACHE_HOME if set)")

    inference_grp = parser.add_argument_group("inference")
    inference_grp.add_argument("--device", default=argparse.SUPPRESS, help="device type to use for PyTorch inference (e.g. cpu, cuda, mps)")
    inference_grp.add_argument("--device_index", default=argparse.SUPPRESS, type=int, help="device index to use for inference")
    inference_grp.add_argument("--batch_size", default=argparse.SUPPRESS, type=int, help="the preferred batch size for inference")
    inference_grp.add_argument("--asr_compute_type", default=argparse.SUPPRESS, type=str, choices=["default", "float16", "float32", "int8"], help="compute type for the ASR model; 'default' uses float16 on GPU, float32 on CPU")
    inference_grp.add_argument("--asr", "-asr", choices=["fast", "quality"], default=argparse.SUPPRESS, help="shorthand for --asr_compute_type (fast=int8, quality=float32); for float16/'default' use --asr_compute_type directly. The granular --asr_compute_type flag always wins if both are given.")
    inference_grp.add_argument("--attn_implementation", default=argparse.SUPPRESS, type=str, choices=["sdpa", "flash_attention_2", "eager"], help="attention implementation for transformer models; 'flash_attention_2' requires flash-attn to be installed")
    inference_grp.add_argument("--threads", type=optional_int, default=argparse.SUPPRESS, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    inference_grp.add_argument("--hf_token", type=str, default=argparse.SUPPRESS, help="Hugging Face Access Token to access PyAnnote gated models")
    inference_grp.add_argument("--compile", action="store_true", default=argparse.SUPPRESS, help="enable torch.compile for the native ASR backend (opt-in; benchmarked to be a net loss by default for this pipeline's variable-length VAD segments — see scripts/bench_asr_compile.py)")

    output_grp = parser.add_argument_group("output")
    output_grp.add_argument("--output_dir", "-o", type=str, default=argparse.SUPPRESS, help="directory to save the outputs")
    output_grp.add_argument("--output_format", "-f", type=str, default=argparse.SUPPRESS, choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    output_grp.add_argument("--verbose", type=str2bool, default=argparse.SUPPRESS, help="whether to print out the progress and debug messages")
    output_grp.add_argument("--log_level", type=str, default=argparse.SUPPRESS, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")
    output_grp.add_argument("--log_file", type=str, default=argparse.SUPPRESS, help="redirect third-party stdout/stderr to this file; cantocaptions_ai log messages are written to both terminal and file")
    output_grp.add_argument("--print_progress", type=str2bool, default=argparse.SUPPRESS, help="if True, display stage progress bars and a timing summary; also enables per-batch progress in transcribe() and align() methods")
    output_grp.add_argument("--vram_checks", type=str2bool, default=argparse.SUPPRESS, help="if True, proactively estimate and log per-stage/per-batch VRAM headroom before running it (queries torch.cuda.mem_get_info each call); set False for zero per-batch overhead when turnaround time matters more than OOM safety margins")
    output_grp.add_argument("--debug_dir", type=str, default=argparse.SUPPRESS, help="if set, write intermediate stage data (audio segments and JSON manifests) to this directory for debugging and replay")
    output_grp.add_argument("--load_debug_dir", type=str, default=argparse.SUPPRESS, help="load intermediate stage data from a previous --debug_dir run; completed stages are skipped and their results are loaded instead")

    audio_grp = parser.add_argument_group("audio")
    audio_grp.add_argument("--audio_start", type=float, default=argparse.SUPPRESS, help="seconds of audio to skip before processing")
    audio_grp.add_argument("--audio_end", type=float, default=argparse.SUPPRESS, help="seconds of audio to cut from the ending")

    vad_grp = parser.add_argument_group("vad")
    vad_grp.add_argument("--vad_method", type=str, default=argparse.SUPPRESS, choices=["pyannote"], help="VAD method to be used")
    vad_grp.add_argument("--vad_onset", type=float, default=argparse.SUPPRESS, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    vad_grp.add_argument("--vad_offset", type=float, default=argparse.SUPPRESS, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    vad_grp.add_argument("--chunk_size", type=int, default=argparse.SUPPRESS, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    isol_grp = parser.add_argument_group("vocal isolation")
    isol_grp.add_argument("--vocal_isolation_method", type=str, default=argparse.SUPPRESS, choices=["none", "mbroformer"], help="vocal isolation method to be used")
    isol_grp.add_argument("--vocal_isolation_batch_size", default=argparse.SUPPRESS, type=int, help="number of fixed-size audio chunks the vocal isolation model processes per batch; larger values give little/no speedup on this model and can regress sharply (benchmark before increasing)")
    isol_grp.add_argument("--vocal_isolation_compute_type", default=argparse.SUPPRESS, type=str, choices=["float32", "float16"], help="compute type (weight dtype) for the vocal isolation model; float16 roughly halves its VRAM usage (falls back to float32 off CUDA)")
    isol_grp.add_argument("--vocal_isolation", "-vi", choices=["fast", "quality"], default=argparse.SUPPRESS, help="shorthand for --vocal_isolation_compute_type (fast=float16, quality=float32). Does NOT affect --vocal_isolation_batch_size (see its own help text: larger batches give little/no speedup on this model and can regress sharply). The granular --vocal_isolation_compute_type flag always wins if both are given.")

    asr_grp = parser.add_argument_group("asr options")
    asr_grp.add_argument("--suppress_tokens", type=str, default=argparse.SUPPRESS, help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    asr_grp.add_argument("--suppress_numerals", action="store_true", default=argparse.SUPPRESS, help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")
    asr_grp.add_argument("--initial_prompt", type=str, default=argparse.SUPPRESS, help="optional text to provide as a prompt for the first window.")
    asr_grp.add_argument("--hotwords", type=str, default=argparse.SUPPRESS, help="hotwords/hint phrases to the model (e.g. \"WhisperX, PyAnnote, GPU\"); improves recognition of rare/technical terms")
    asr_grp.add_argument("--condition_on_previous_text", type=str2bool, default=argparse.SUPPRESS, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    asr_grp.add_argument("--fp16", type=str2bool, default=argparse.SUPPRESS, help="whether to perform inference in fp16; True by default")

    ensemble_grp = parser.add_argument_group("ensemble & LLM correction")
    ensemble_grp.add_argument("--ensemble_model", type=str, default=argparse.SUPPRESS, choices=["none", "whisper"], help="second ASR model for ensemble correction; 'whisper' runs alvanlii/whisper-small-cantonese via faster-whisper alongside the primary model (requires pip install 'cantocaptions_ai[ensemble]')")
    ensemble_grp.add_argument("--llm_correction", action="store_true", default=argparse.SUPPRESS, help="run LLM-based per-segment particle correction and full-document name normalization after transcription")
    ensemble_grp.add_argument("--llm_model", type=str, default=argparse.SUPPRESS, help="HuggingFace model ID for LLM correction (used with --llm_correction)")
    ensemble_grp.add_argument("--llm_model_dir", type=str, default=argparse.SUPPRESS, help="local path to LLM weights; uses HF cache if not set")
    ensemble_grp.add_argument("--reference_subtitle", type=str, default=argparse.SUPPRESS, metavar="SUBTITLE_FILE", help="standard Chinese subtitle file (SRT/VTT) used as reference for LLM correction; fixes homophone errors in proper nouns, idioms, etc. (requires --llm_correction)")
    ensemble_grp.add_argument("--reference_correction_semantic", action="store_true", default=argparse.SUPPRESS, help="also attempt semantic fixes from the reference subtitle (e.g. missing negations, punctuation); higher false-positive risk, requires --reference_subtitle")

    align_grp = parser.add_argument_group("alignment")
    align_grp.add_argument("--align_model", default=argparse.SUPPRESS, help="Name of phoneme-level ASR model to do alignment")
    align_grp.add_argument("--interpolate_method", default=argparse.SUPPRESS, choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    align_grp.add_argument("--no_align", action='store_true', default=argparse.SUPPRESS, help="Do not perform phoneme alignment")
    align_grp.add_argument("--return_char_alignments", action='store_true', default=argparse.SUPPRESS, help="Return character-level alignments in the output json file")
    align_grp.add_argument("--align_padding", type=float, default=argparse.SUPPRESS, help="The minimum allowed timebetween subttitles.")
    align_grp.add_argument("--align_release", type=float, default=argparse.SUPPRESS, help="When aligning the end of an utterance, add this duration to the end as additional release time.")
    align_grp.add_argument("--align_merge_distance", type=float, default=argparse.SUPPRESS, help="The maximum distance between utterances that allows them to be merged.")
    align_grp.add_argument("--align_batch_size", default=argparse.SUPPRESS, type=int, help="number of VAD segments the alignment model processes per batch")
    align_grp.add_argument("--align_compute_type", default=argparse.SUPPRESS, type=str, choices=["float32", "float16"], help="compute type (weight dtype) for the alignment model; float16 lowers VRAM usage but may reduce forced-alignment accuracy (falls back to float32 off CUDA)")
    align_grp.add_argument("--align", "-a", choices=["fast", "quality"], default=argparse.SUPPRESS, help="shorthand for --align_compute_type (fast=float16, quality=float32). Does NOT affect --align_batch_size (no benchmarked safe bump exists for the 'fast' tier — see scripts/bench_alignment_batching.py). The granular --align_compute_type flag always wins if both are given.")

    subtitle_grp = parser.add_argument_group("subtitle formatting")
    subtitle_grp.add_argument("--max_line_width", type=optional_int, default=argparse.SUPPRESS, help="(not possible with --no_align) the maximum number of characters in a line before text cleaning breaks the line")
    subtitle_grp.add_argument("--max_line_count", type=optional_int, default=argparse.SUPPRESS, help="(not possible with --no_align) the maximum number of lines in a segment; text cleaning only breaks lines when this is 2 or more")
    subtitle_grp.add_argument("--highlight_words", type=str2bool, default=argparse.SUPPRESS, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    subtitle_grp.add_argument("--segment_resolution", type=str, default=argparse.SUPPRESS, choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    clean_grp = parser.add_argument_group("text cleaning")
    clean_grp.add_argument("--no_clean_text", action="store_true", default=argparse.SUPPRESS, help="disable Cantonese subtitle text cleaning (punctuation, HK conventions, particle fixes, interjection removal, line breaking)")
    clean_grp.add_argument("--clean_rules_dir", type=str, default=argparse.SUPPRESS, metavar="DIR", help="directory containing pipeline.toml and rule .toml files overriding the built-in cleaning rules")

    diarize_grp = parser.add_argument_group("diarization")
    diarize_grp.add_argument("--diarize", action="store_true", default=argparse.SUPPRESS, help="Apply diarization to assign speaker labels to each segment/word")
    diarize_grp.add_argument("--diarize_merge", action="store_true", default=argparse.SUPPRESS, help="Use diarization to detect the same speaker on consecutive lines and merge them.")
    diarize_grp.add_argument("--min_speakers", default=argparse.SUPPRESS, type=int, help="Minimum number of speakers to in audio file")
    diarize_grp.add_argument("--max_speakers", default=argparse.SUPPRESS, type=int, help="Maximum number of speakers to in audio file")
    diarize_grp.add_argument("--diarize_model", default=argparse.SUPPRESS, type=str, help="Name of the speaker diarization model to use")
    diarize_grp.add_argument("--speaker_embeddings", action="store_true", default=argparse.SUPPRESS, help="Include speaker embeddings in JSON output (only works with --diarize)")

    verify_grp = parser.add_argument_group("speaker verification")
    verify_grp.add_argument("--verify_speakers", action="store_true", default=argparse.SUPPRESS, help="Apply speaker verification to split out segments where speakers don't match")

    input_grp = parser.add_argument_group("batch input")
    input_grp.add_argument("--input_dir", type=str, default=argparse.SUPPRESS, metavar="DIR", help="directory of media files to transcribe (mutually exclusive with positional audio args)")
    input_grp.add_argument("--recursive", action="store_true", default=argparse.SUPPRESS, help="when used with --input_dir, also scan subdirectories for media files")
    # fmt: on
    return parser


def cli():
    parser = build_parser()
    explicit = vars(parser.parse_args())  # only user-typed keys (+ positional audio)

    merged = resolve_pipeline_args(parser, explicit)  # dataclass defaults -> cfg file -> presets -> explicit

    log_level = explicit.get("log_level")
    verbose = merged.get("verbose")
    log_file = merged.pop("log_file", None)

    if log_level is not None:
        setup_logging(level=log_level, log_file=log_file)
    elif verbose:
        setup_logging(level="info", log_file=log_file)
    else:
        setup_logging(level="warning", log_file=log_file)

    logger.debug(
        "Resolved pipeline config (cfg=%s): %s",
        explicit.get("cfg", "default"),
        {k: v for k, v in sorted(merged.items()) if k != "hf_token"},
    )

    audio = merged.get("audio") or []
    input_dir = merged.pop("input_dir", None)
    recursive = merged.pop("recursive", False)
    if audio and input_dir:
        parser.error("positional audio files and --input_dir are mutually exclusive")
    if not audio and not input_dir:
        parser.error("provide at least one audio file or --input_dir")
    if input_dir:
        try:
            audio = discover_media_files(input_dir, recursive=recursive)
        except ValueError as e:
            parser.error(str(e))
    merged["audio"] = audio

    from cantocaptions_ai.pipeline.transcribe import transcribe_task

    transcribe_task(merged, parser)

if __name__ == "__main__":
    cli()
