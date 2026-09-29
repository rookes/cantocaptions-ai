"""Stand-ins for the pipeline's models, so the whole pipeline runs on CPU in seconds.

The trick that makes alignment deterministic: the synthetic audio *encodes its own
transcript*. Every 40 ms frame of speech holds a constant sample value that names a token
of the fake align model's vocabulary, so the fake CTC model can read the token straight off
the audio and emit a one-hot emission for it. Forced alignment then has exactly one path to
find, and every character's time is known in advance from the script.

The encoding is peaky, like a real CTC model: a character fires on one frame and blank
fills the rest of its slot. The aligner's trellis can only "stay" on a token through blank,
so a character held for several frames would be squeezed onto consecutive single frames.

    ScriptedAudio  -- the script -> WAV samples, align dictionary and expected char times
    StubVad        -- splits the audio at digital silence (or tiles it for --realign)
    StubAsr        -- returns the scripted text for each VAD segment
    FakeAligner    -- processor + CTC model reading tokens back out of the audio
    install(...)   -- monkeypatches the three loaders transcribe.py imports lazily
"""
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from cantocaptions_ai.pipeline.align_profiles import DEFAULT_ALIGN_PROFILE
from cantocaptions_ai.pipeline.align_vocab import LEVEL_OFF, VocabRepair
from cantocaptions_ai.pipeline.asr import QwenPipeline
from cantocaptions_ai.pipeline.vad import VadProcessor

SR = 16000
FRAME = 640                  # samples per emission frame: 25 fps, like wav2vec2-BERT
FRAME_S = FRAME / SR
CHAR_FRAMES = 4              # frames per character: one peak, then blank
PAUSE_FRAMES = 5             # blank frames a punctuation mark stands for
BLANK = "<pad>"
PUNCTUATION = set("，。？！；…,.?!;")
WORD_DELIMITER = "|"  # what a space is spoken as, like a real wav2vec2 CTC vocabulary

# Three utterances of plain written Cantonese the default cleaning manifest leaves alone
# (no numerals, no clause-comma triggers, no interjection-only lines).
DEFAULT_SCRIPT: Tuple[Tuple[float, str], ...] = (
    (1.0, "你好，今日天氣幾好。"),
    (4.0, "我哋去公園行下啦。"),
    (7.0, "好呀，我帶埋隻狗。"),
)


def _level(token_id: int) -> float:
    # id -> sample value; 0 stays reserved for silence, which is what the stub VAD cuts on.
    return (token_id + 1) / 100.0


def _token(level: float) -> int:
    return int(round(level * 100.0)) - 1


@dataclass
class ScriptedAudio:
    script: Sequence[Tuple[float, str]] = DEFAULT_SCRIPT
    duration: float = 10.0
    # Tokens the align model knows but the audio never says (e.g. spot-check alternatives).
    extra_vocab: Sequence[str] = ()
    dictionary: Dict[str, int] = field(init=False)
    samples: np.ndarray = field(init=False)
    char_times: List[List[Tuple[str, float, float]]] = field(init=False)  # per utterance

    def __post_init__(self):
        chars = sorted({self._token_char(c) for _, text in self.script for c in text
                        if c not in PUNCTUATION} | set(self.extra_vocab))
        self.dictionary = {BLANK: 0, **{c: i + 1 for i, c in enumerate(chars)}}
        audio = np.zeros(int(self.duration * SR), dtype=np.float32)
        self.char_times = []
        for start, text in self.script:
            assert round(start / FRAME_S, 6).is_integer(), "utterances must start on a frame"
            frame = int(round(start / FRAME_S))
            times = []
            for c in text:
                if c in PUNCTUATION:
                    ids = [0] * PAUSE_FRAMES
                else:
                    ids = [self.dictionary[self._token_char(c)]] + [0] * (CHAR_FRAMES - 1)
                    if c != " ":
                        times.append((c, frame * FRAME_S, (frame + 1) * FRAME_S))
                for token in ids:
                    audio[frame * FRAME:(frame + 1) * FRAME] = _level(token)
                    frame += 1
            self.char_times.append(times)
        self.samples = audio

    @staticmethod
    def _token_char(c: str) -> str:
        return WORD_DELIMITER if c == " " else c.lower()

    def utterance_span(self, i: int) -> Tuple[float, float]:
        return self.char_times[i][0][1], self.char_times[i][-1][2]

    def text_at(self, start: float, end: float) -> str:
        """The scripted text of every utterance inside [start, end)."""
        return "".join(t for s, t in self.script if start - 1e-6 <= s < end)

    def write_wav(self, path) -> str:
        pcm = np.round(self.samples * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        return str(path)


# --- VAD ------------------------------------------------------------------------------

class StubVad(VadProcessor):
    """Splits at runs of digital silence; with cover_all, tiles the file like --realign wants."""

    calls = 0

    def __init__(self, cover_all: bool = False, min_gap: float = 0.3):
        self.cover_all = cover_all
        self.min_gap = min_gap

    def process(self, input, *, progress_callback=None):
        type(self).calls += 1
        audio = np.asarray(input, dtype=np.float32)
        voiced = np.flatnonzero(np.abs(audio) > 1e-4)
        regions = []
        if len(voiced):
            breaks = np.flatnonzero(np.diff(voiced) > self.min_gap * SR)
            starts = np.concatenate([[voiced[0]], voiced[breaks + 1]])
            ends = np.concatenate([voiced[breaks] + 1, [voiced[-1] + 1]])
            regions = list(zip(starts.tolist(), ends.tolist()))
        if self.cover_all:
            # Contiguous chunks cut halfway through each silence.
            cuts = [0] + [(e + s) // 2 for (_, e), (s, _) in zip(regions, regions[1:])] + [len(audio)]
            regions = list(zip(cuts, cuts[1:]))
        return [
            {"start": s / SR, "end": e / SR, "audio": audio[s:e].copy()} for s, e in regions
        ]


# --- ASR ------------------------------------------------------------------------------

class StubAsr(QwenPipeline):
    """Transcribes each VAD segment as whatever the script says is spoken inside it."""

    calls = 0

    def __init__(self, scripted: ScriptedAudio, offset: float = 0.0):
        self.scripted = scripted
        self.offset = offset  # clip start, when the pipeline was given --audio_start

    def process(self, input, *, progress_callback=None):
        type(self).calls += 1
        segments = [
            {"start": seg["start"], "end": seg["end"],
             "text": self.scripted.text_at(seg["start"] + self.offset, seg["end"] + self.offset)}
            for seg in input
        ]
        return {"segments": segments, "language": "Cantonese"}


# --- Alignment -----------------------------------------------------------------------

def fake_processor(wavs, sampling_rate=SR, return_tensors="pt", return_attention_mask=True,
                   padding=True):
    """One feature per 40 ms frame: the frame's mean sample value (the encoded token)."""
    if isinstance(wavs, np.ndarray) and wavs.ndim == 1:
        wavs = [wavs]
    frames = [np.asarray(w, dtype=np.float32)[: len(w) // FRAME * FRAME].reshape(-1, FRAME).mean(1)
              for w in wavs]
    longest = max((len(f) for f in frames), default=0)
    features = torch.zeros(len(frames), longest, 1)
    mask = torch.zeros(len(frames), longest, dtype=torch.long)
    for row, f in enumerate(frames):
        features[row, :len(f), 0] = torch.from_numpy(f)
        mask[row, :len(f)] = 1
    return {"input_features": features, "attention_mask": mask}


class _Output:
    def __init__(self, logits):
        self.logits = logits


class FakeCtcModel:
    """Emits the token each frame's audio encodes, with near-certainty."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def parameters(self):
        yield torch.zeros(1)

    def _get_feat_extract_output_lengths(self, lengths, add_adapter=None):
        return lengths

    def __call__(self, input_features, attention_mask=None):
        ids = torch.round(input_features[..., 0] * 100).long() - 1
        ids = ids.clamp(0, self.vocab_size - 1)
        logits = torch.full((*ids.shape, self.vocab_size), -20.0)
        logits.scatter_(-1, ids.unsqueeze(-1), 20.0)
        return _Output(logits)


def fake_align_model(scripted: ScriptedAudio, language: str = "yue"):
    dictionary = dict(scripted.dictionary)
    return FakeCtcModel(len(dictionary)), {
        "language": language,
        "dictionary": dictionary,
        "type": "huggingface",
        "processor": fake_processor,
        "frame_rate": 1.0 / FRAME_S,
        "vocab_repair": VocabRepair(dictionary, LEVEL_OFF),
        "profile": DEFAULT_ALIGN_PROFILE,
    }


# --- Wiring --------------------------------------------------------------------------

def install(monkeypatch, scripted: ScriptedAudio, asr_offset: float = 0.0):
    """Point the loaders transcribe.py imports at call time to the fakes above."""
    from cantocaptions_ai.pipeline import alignment, asr, vad

    StubVad.calls = 0
    StubAsr.calls = 0
    monkeypatch.setattr(vad, "load_vad", lambda **kw: StubVad(cover_all=kw.get("cover_all", False)))
    monkeypatch.setattr(asr, "load_model", lambda *a, **kw: StubAsr(scripted, asr_offset))
    monkeypatch.setattr(alignment, "load_align_model",
                        lambda *a, **kw: fake_align_model(scripted))
