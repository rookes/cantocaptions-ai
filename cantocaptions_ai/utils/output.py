import io
import json
import os
import re
import sys
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, TextIO, List

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

LANGUAGES_WITHOUT_SPACES = ["ja", "zh", "yue"]

# Mapping of language codes to NLTK Punkt tokenizer model names
PUNKT_LANGUAGES = {
    'cs': 'czech',
    'da': 'danish',
    'de': 'german',
    'el': 'greek',
    'en': 'english',
    'es': 'spanish',
    'et': 'estonian',
    'fi': 'finnish',
    'fr': 'french',
    'it': 'italian',
    'nl': 'dutch',
    'no': 'norwegian',
    'pl': 'polish',
    'pt': 'portuguese',
    'sl': 'slovene',
    'sv': 'swedish',
    'tr': 'turkish',
    "ml": "malayalam",
    "ru": "russian",
}

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default encoding with an '?',
        # avoiding UnicodeEncodeError (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding, errors="replace").decode(system_encoding)

else:

    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the round-trip encoding
        return string


def exact_div(x, y):
    assert x % y == 0
    return x // y


def str2bool(string):
    str2val = {"True": True, "False": False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def _with_speaker(
    segment: dict, text: str, options: dict, fmt: str = "[{speaker}]: {text}"
) -> str:
    """Prefix *text* with the segment's speaker label, when the caller asked for labels.

    Diarization is normally run to keep cue assembly from merging across a speaker change,
    not to caption who is talking, so labels stay out of the subtitle text unless
    ``--speaker_labels`` opts in. Segments diarization could not confidently attribute are
    always left bare rather than captioned as an unknown speaker.
    """
    speaker = segment.get("speaker")
    if not options.get("speaker_labels") or speaker is None:
        return text
    return fmt.format(speaker=speaker, text=text)


def output_names(paths: Iterable[str], input_dir: Optional[str] = None) -> Dict[str, str]:
    """The name each input's outputs and debug checkpoints are written under.

    A name is the file's stem, or with ``input_dir`` its path relative to that directory
    without the extension (``s1/ep01`` for ``input_dir/s1/ep01.mkv``), always with ``/``
    separators. Mirroring the input tree is what lets a recursive run hold two
    ``ep01.mkv`` from different seasons: keyed by stem they would overwrite each other's
    subtitles and share one debug cache.

    Raises ConfigError if two inputs would still share a name (two files passed directly
    from different folders, or ``a.mkv`` beside ``a.mp4``).
    """
    from cantocaptions_ai.errors import ConfigError

    names: Dict[str, str] = {}
    for path in paths:
        if input_dir is not None:
            rel = Path(os.path.relpath(path, input_dir))
            name = rel.with_name(rel.stem.strip()).as_posix()
        else:
            name = Path(path).stem.strip()
        names[path] = name

    by_name: Dict[str, List[str]] = {}
    for path, name in names.items():
        by_name.setdefault(name, []).append(path)
    clashes = {name: ps for name, ps in by_name.items() if len(ps) > 1}
    if clashes:
        detail = "; ".join(f"{name!r}: {', '.join(ps)}" for name, ps in sorted(clashes.items()))
        hint = "" if input_dir is not None else " (pass their parent folder as --input_dir instead)"
        raise ConfigError(f"Inputs would overwrite each other's output{hint}: {detail}")
    return names


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, name: str, options: dict):
        """Write *result* to ``output_dir/<name>.<extension>``.

        ``name`` comes from :func:`output_names`, and may hold ``/`` for a mirrored
        subfolder, which is created as needed.
        """
        output_path = os.path.join(self.output_dir, *name.split("/")) + "." + self.extension
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options)

    def write_result(self, result: dict, file: TextIO, options: dict):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO, options: dict):
        for segment in result["segments"]:
            text = segment["text"].strip()
            print(_with_speaker(segment, text, options), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(self, result: dict, options: dict):
        if len(result["segments"]) == 0:
            return

        for segment in result["segments"]:
            segment_start = self.format_timestamp(segment["start"])
            segment_end = self.format_timestamp(segment["end"])
            segment_text = segment["text"].strip().replace("-->", "->")
            yield (segment_start, segment_end, _with_speaker(segment, segment_text, options),
                   segment.get("style_tags") or "")

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(self, result: dict, file: TextIO, options: dict):
        from cantocaptions_ai.utils.subtitles import is_top_position

        print("WEBVTT\n", file=file)
        for start, end, text, style in self.iterate_result(result, options):
            # WebVTT has no override blocks; its own way of saying "top of the frame" is a
            # cue setting. Any other ASS styling has no equivalent and is dropped.
            settings = " line:0" if is_top_position(style) else ""
            print(f"{start} --> {end}{settings}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(self, result: dict, file: TextIO, options: dict):
        for i, (start, end, text, style) in enumerate(
            self.iterate_result(result, options), start=1
        ):
            # The override blocks go back exactly as the input had them (``{\an8}``); players
            # that read SRT honour them, and the rest show the text as before.
            print(f"{i}\n{start} --> {end}\n{style}{text}\n", file=file, flush=True)


def _single_line(text: str) -> str:
    """A cue's text as one row: tabs to spaces, line breaks removed.

    TSV and Audacity labels are one row per cue, so the break the line-layout step put into
    a two-line cue would otherwise split the row in two. The break is layout only -- the
    current (CJK) line breaker inserts it without removing anything -- so dropping it
    restores the text. A script that breaks on spaces will need a space here instead.
    """
    return "".join(text.strip().replace("\t", " ").splitlines())


class WriteTSV(ResultWriter):
    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(_single_line(segment["text"]), file=file, flush=True)

class WriteAudacity(ResultWriter):
    extension: str = "aud"

    def write_result(self, result: dict, file: TextIO, options: dict):
        ARROW = "	"
        for segment in result["segments"]:
            print(segment["start"], file=file, end=ARROW)
            print(segment["end"], file=file, end=ARROW)
            text = _single_line(segment["text"])
            print(_with_speaker(segment, text, options, fmt="[[{speaker}]]{text}"), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO, options: dict):
        json.dump(result, file, ensure_ascii=False)


def writer_args(cfg) -> dict:
    """The writer ``options`` dict for a PipelineConfig.

    The one place both output paths build it from -- the CLI's file writers and the
    service's in-memory ``render_result`` -- so the two cannot drift apart.
    """
    return {
        "max_line_count": cfg.max_line_count,
        "max_line_width": cfg.max_line_width,
        "speaker_labels": cfg.speaker_labels,
    }


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, str, dict], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
    }
    optional_writers = {
        "aud": WriteAudacity,
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, file: str, options: dict):
            for writer in all_writers:
                writer(result, file, options)

        return write_all

    if output_format in optional_writers:
        return optional_writers[output_format](output_dir)
    return writers[output_format](output_dir)


# Formats that render to a single text document (i.e. everything except "all", which
# fans out to multiple files and so has no single-string form).
_SINGLE_DOC_WRITERS = {
    "txt": WriteTXT,
    "vtt": WriteVTT,
    "srt": WriteSRT,
    "tsv": WriteTSV,
    "json": WriteJSON,
    "aud": WriteAudacity,
}


def render_result(result: dict, output_format: str, options: Optional[dict] = None) -> str:
    """Render a transcription ``result`` to a subtitle/transcript string in memory.

    The disk-free counterpart to ``get_writer``: reuses each writer's
    ``write_result(result, file, options)`` (which already accepts any TextIO) by
    pointing it at an ``io.StringIO``. Lets a server serve output as an HTTP body
    without writing to ``output_dir``.

    ``"all"`` is rejected — it produces several files, not one string; callers that
    want multiple formats should call this once per format.
    """
    if output_format == "all":
        raise ValueError("render_result cannot render output_format='all' to a single string")
    writer_cls = _SINGLE_DOC_WRITERS.get(output_format)
    if writer_cls is None:
        raise ValueError(f"Unknown output_format: {output_format!r}")
    buf = io.StringIO()
    # output_dir is unused by write_result (only ResultWriter.__call__ touches disk).
    writer_cls(output_dir="").write_result(result, file=buf, options=options or {})
    return buf.getvalue()

