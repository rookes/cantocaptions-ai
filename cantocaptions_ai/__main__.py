import argparse
import importlib.metadata
import platform

import torch

from cantocaptions_ai.utils.output import (LANGUAGES, TO_LANGUAGE_CODE,
                            optional_int, str2bool)
from cantocaptions_ai.utils.log_utils import setup_logging

def cli():
    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--language", type=str, default="yue", choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")
    parser.add_argument("--retime", type=str, default=None, metavar="SUBTITLE_FILE", help="subtitle file to retime against the audio (SRT, VTT, etc.). Skips ASR; updates timings only, text is preserved.")
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {importlib.metadata.version('cantocaptions-ai')}", help="Show cantocaptions-ai version information and exit")
    parser.add_argument("--python-version", "-P", action="version", version=f"Python {platform.python_version()} ({platform.python_implementation()})", help="Show python version information and exit")

    model_grp = parser.add_argument_group("model")
    model_grp.add_argument("--model", default="Qwen3-ASR", help="name of the model to use")
    model_grp.add_argument("--model_cache_only", type=str2bool, default=False, help="If True, will not attempt to download models, instead using cached models from --model_dir")
    model_grp.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")

    inference_grp = parser.add_argument_group("inference")
    inference_grp.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"), help="device type to use for PyTorch inference (e.g. cpu, cuda, mps)")
    inference_grp.add_argument("--device_index", default=0, type=int, help="device index to use for inference")
    inference_grp.add_argument("--batch_size", default=28, type=int, help="the preferred batch size for inference")
    inference_grp.add_argument("--compute_type", default="default", type=str, choices=["default", "float16", "float32", "int8"], help="compute type for computation; 'default' uses float16 on GPU, float32 on CPU")
    inference_grp.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    inference_grp.add_argument("--hf_token", type=str, default=None, help="Hugging Face Access Token to access PyAnnote gated models")

    output_grp = parser.add_argument_group("output")
    output_grp.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    output_grp.add_argument("--output_format", "-f", type=str, default="srt", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="format of the output file; if not specified, all available formats will be produced")
    output_grp.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")
    output_grp.add_argument("--log-level", type=str, default=None, choices=["debug", "info", "warning", "error", "critical"], help="logging level (overrides --verbose if set)")
    output_grp.add_argument("--log_file", type=str, default=None, help="redirect third-party stdout/stderr to this file; cantocaptions_ai log messages are written to both terminal and file")
    output_grp.add_argument("--print_progress", type=str2bool, default=True, help="if True, display stage progress bars and a timing summary; also enables per-batch progress in transcribe() and align() methods")
    output_grp.add_argument("--debug_dir", type=str, default=None, help="if set, write intermediate stage data (audio segments and JSON manifests) to this directory for debugging and replay")
    output_grp.add_argument("--load_debug_dir", type=str, default=None, help="load intermediate stage data from a previous --debug_dir run; completed stages are skipped and their results are loaded instead")

    audio_grp = parser.add_argument_group("audio")
    audio_grp.add_argument("--audio_start", type=float, default=None, help="seconds of audio to skip before processing")
    audio_grp.add_argument("--audio_end", type=float, default=None, help="seconds of audio to cut from the ending")

    vad_grp = parser.add_argument_group("vad")
    vad_grp.add_argument("--vad_method", type=str, default="pyannote", choices=["pyannote"], help="VAD method to be used")
    vad_grp.add_argument("--vad_onset", type=float, default=0.500, help="Onset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected")
    vad_grp.add_argument("--vad_offset", type=float, default=0.363, help="Offset threshold for VAD (see pyannote.audio), reduce this if speech is not being detected.")
    vad_grp.add_argument("--chunk_size", type=int, default=30, help="Chunk size for merging VAD segments. Default is 30, reduce this if the chunk is too long.")

    isol_grp = parser.add_argument_group("vocal isolation")
    isol_grp.add_argument("--vocal_isolation_method", type=str, default="mbroformer", choices=["none", "mbroformer"], help="vocal isolation method to be used")

    asr_grp = parser.add_argument_group("asr options")
    asr_grp.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    asr_grp.add_argument("--suppress_numerals", action="store_true", help="whether to suppress numeric symbols and currency symbols during sampling, since wav2vec2 cannot align them correctly")
    asr_grp.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    asr_grp.add_argument("--hotwords", type=str, default=None, help="hotwords/hint phrases to the model (e.g. \"WhisperX, PyAnnote, GPU\"); improves recognition of rare/technical terms")
    asr_grp.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    asr_grp.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    ensemble_grp = parser.add_argument_group("ensemble & LLM correction")
    ensemble_grp.add_argument("--ensemble_model", type=str, default="none", choices=["none", "whisper"], help="second ASR model for ensemble correction; 'whisper' runs alvanlii/whisper-small-cantonese via faster-whisper alongside the primary model (requires pip install 'cantocaptions_ai[ensemble]')")
    ensemble_grp.add_argument("--llm_correction", action="store_true", help="run LLM-based per-segment particle correction and full-document name normalization after transcription")
    ensemble_grp.add_argument("--llm_model", type=str, default="Qwen/Qwen3-4B", help="HuggingFace model ID for LLM correction (used with --llm_correction)")
    ensemble_grp.add_argument("--llm_model_dir", type=str, default=None, help="local path to LLM weights; uses HF cache if not set")

    align_grp = parser.add_argument_group("alignment")
    align_grp.add_argument("--align_model", default=None, help="Name of phoneme-level ASR model to do alignment")
    align_grp.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="For word .srt, method to assign timestamps to non-aligned words, or merge them into neighbouring.")
    align_grp.add_argument("--no_align", action='store_true', help="Do not perform phoneme alignment")
    align_grp.add_argument("--return_char_alignments", action='store_true', help="Return character-level alignments in the output json file")
    align_grp.add_argument("--align_padding", type=float, default=0.04, help="The minimum allowed timebetween subttitles.")
    align_grp.add_argument("--align_release", type=float, default=0.45, help="When aligning the end of an utterance, add this duration to the end as additional release time.")
    align_grp.add_argument("--align_merge_distance", type=float, default=0.12, help="The maximum distance between utterances that allows them to be merged.")

    subtitle_grp = parser.add_argument_group("subtitle formatting")
    subtitle_grp.add_argument("--max_line_width", type=optional_int, default=18, help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")
    subtitle_grp.add_argument("--max_line_count", type=optional_int, default=1, help="(not possible with --no_align) the maximum number of lines in a segment")
    subtitle_grp.add_argument("--highlight_words", type=str2bool, default=False, help="(not possible with --no_align) underline each word as it is spoken in srt and vtt")
    subtitle_grp.add_argument("--segment_resolution", type=str, default="sentence", choices=["sentence", "chunk"], help="(not possible with --no_align) the maximum number of characters in a line before breaking the line")

    diarize_grp = parser.add_argument_group("diarization")
    diarize_grp.add_argument("--diarize", action="store_true", help="Apply diarization to assign speaker labels to each segment/word")
    diarize_grp.add_argument("--diarize_merge", action="store_true", help="Use diarization to detect the same speaker on consecutive lines and merge them.")
    diarize_grp.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers to in audio file")
    diarize_grp.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers to in audio file")
    diarize_grp.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1", type=str, help="Name of the speaker diarization model to use")
    diarize_grp.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output (only works with --diarize)")

    verify_grp = parser.add_argument_group("speaker verification")
    verify_grp.add_argument("--verify_speakers", action="store_true", help="Apply speaker verification to split out segments where speakers don't match")
    # fmt: on

    args = parser.parse_args().__dict__

    log_level = args.get("log_level")
    verbose = args.get("verbose")
    log_file = args.pop("log_file", None)

    if log_level is not None:
        setup_logging(level=log_level, log_file=log_file)
    elif verbose:
        setup_logging(level="info", log_file=log_file)
    else:
        setup_logging(level="warning", log_file=log_file)

    from cantocaptions_ai.pipeline.transcribe import transcribe_task

    transcribe_task(args, parser)

if __name__ == "__main__":
    cli()
