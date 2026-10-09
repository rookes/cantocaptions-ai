# Adding a language

Everything the pipeline knows about a language lives in one **language pack**: a `LanguagePack`
registered under the language's code (`cantocaptions_ai/languages/`). Cantonese (`languages/yue/`) is
the reference pack. This guide builds one up from the minimum.

## What you get without a pack

Any language code the pipeline accepts already has a *generic* pack (`languages/base.py`,
`generic_pack`):

- script and punctuation follow from the code: CJK for `zh`/`ja`/`yue`, space-separated with Latin
  punctuation otherwise;
- the built-in align model for the language, if `languages/align_defaults.py` has one;
- no default ASR model and no cleaning rules.

That is enough to run the pipeline **raw**: `validate_config` asks for `--model` and `--no_clean_text`
(and `--align_model` if there is no built-in one), then the audio is transcribed, aligned, split into
cues and word-wrapped.

A pack exists to remove those flags and to encode what "good subtitles" means for the language.

## The minimal pack

```python
# cantocaptions_ai/languages/en/__init__.py
from cantocaptions_ai.languages.base import LanguagePack
from cantocaptions_ai.text_profiles import LATIN_PUNCTUATION, SPACED_SCRIPT

EN = LanguagePack(
    code="en",
    script=SPACED_SCRIPT,          # words joined with spaces, 42-char lines, word-wrapped
    punctuation=LATIN_PUNCTUATION, # sentence splits at . ? ! ; and a comma lets cues join
    default_model="whisper-large-v3",  # any MODEL_PROFILES key, or a hub id / path
    default_align_model="facebook/wav2vec2-base-960h",
)
```

Register it next to yue in `languages/__init__.py` (`_register_builtin_packs`), from your own code
with `register_language_pack(EN)`, or from a package of its own (see "Shipping a pack as its own package"). With only this, `--language en` needs no `--model`, but still needs
`--no_clean_text`: `fully_supported` means a default model **and** cleaning rules.

The model can be any Qwen3-ASR, Whisper or wav2vec2-family CTC checkpoint: the backend is read from
its `config.json` (`pipeline/asr.py`, `backend_for`). A family with no backend yet needs a
`BatchedAsrStage` subclass (one `_infer_batch` method; see `pipeline/_asr_whisper.py`) registered in
`ASR_BACKENDS`.

If the ASR model is trained for particular languages only, list them on its `ModelProfile`
(`pipeline/model_profiles.py`, `languages=frozenset({...})`). `validate_config` then refuses it for any
other language.

## Optional parts

| Field | What it buys |
|---|---|
| `script` | `ScriptConfig(word_separator, line_width, layout)`. `layout` names a line breaker in `cleaning/layout.py` (`"cjk"`, `"word"`); add one there if neither fits. |
| `punctuation` | `PunctuationConfig(split_chars, mergeable_chars)`: where alignment splits clauses and which trailing marks still let two cues join. |
| `conventions` | `{model name: ModelConventions}`: how one ASR model writes this language. `normalization` (e.g. OpenCC), `spotchecks` (interchangeable characters alignment may swap for the one the audio supports), `segmentation.leading_markers` (clause-leading words cue assembly rejoins forwards), `cleaning_manifest`, and `punctuation`/`script` overrides. Unlisted models get the pack defaults. |
| `cleaning` | `CleaningSpec(rules_dir, manifest, builtin_steps, noise_tokens)`. See below. |
| `track_selector` | `streams -> index`, when the audio-track tags need more than a language-code match (yue prefers an explicit Cantonese track, then any Chinese one). |
| `correction_prompts` | `CorrectionPrompts` for `--llm_correction`. Without it, the option is refused for the language. |
| `proofreading` / `default_proofreading` | `{name: ProofreadStandard}` for `--proofread`, and which one a run uses unless `--proofread_standard` names another. A standard is data, not code: what the written language is called, a conventions Markdown file, a few bullets of the errors ASR typically makes in it, an example of how one person's names vary, and a `foreign_register` of characters or words an edit may not introduce (for Cantonese, Standard Written Chinese forms the model would otherwise copy from a reference subtitle), and `final_particles` with the two prompt bullets for them, if the language has sentence-final particles a text-only model cannot tell apart (changes to them are flagged, not applied, unless `--proofread_particles allow`). A language may ship several standards. Without one, `--proofread` still runs, correcting for meaning only. |
| `ensemble_model` | `(hub repo, CTranslate2 subfolder)` of a faster-whisper second opinion for `--ensemble_model`. |
| `char_readings` | A factory returning a `CharReadings` (`languages/base.py`): each character's reading, the reading without its tone, a variant character's standard form, and corpus frequencies. Alignment uses it to give a character the align model has no token for the token of a variant or homophone it does have (`--align_char_substitution`). Without it, only the align model's own substitution table applies. Cantonese's is `languages/yue/readings.py`. Make it a factory that imports lazily, so reading the registry stays cheap. |

The align model can be any Hugging Face CTC checkpoint (wav2vec2, wav2vec2-BERT, HuBERT, WavLM, ...) or a
torchaudio pipeline bundle name; `pipeline/align_backends.py` tells them apart from the model itself.
Align-model behaviour (an audio primer, hand-picked substitutions, the character-substitution level) is
per *align model*, not per language: add an `AlignProfile` in `pipeline/align_profiles.py`.

## Cleaning rules

`CleaningSpec.rules_dir` holds one or more manifests plus the rule files they name. A manifest is TOML:

```toml
[[pre_align]]            # optional: runs on raw ASR text, before alignment
type = "rules"
file = "clauses.toml"

[[steps]]                # runs on each finished cue, in order
type = "rules"
file = "punctuation.toml"

[[steps]]
type = "builtin"
name = "linebreak"       # always available: the script's line breaker at the configured width
```

A rule file is an ordered list of regex substitutions:

```toml
[[rules]]
pattern = "\\bgonna\\b"
replace = "going to"
comment = "optional"
```

`builtin_steps` is a zero-argument function returning `{name: text -> text}` for coded steps a manifest
may name (yue's are numerals, question particles, acronyms and trimming). It is called only when a
cleaner is built, so heavy NLP imports stay out of the registry. `noise_tokens` are whole cues dropped as
pure interjection. `--clean_rules_dir` lets a user swap in their own directory.

## Shipping a pack as its own package

A pack does not have to live in this repository. Any installed distribution can register one through
the `cantocaptions_ai.languages` entry-point group; the registry picks it up the first time it is read,
with no import or registration call needed:

```toml
# pyproject.toml of your package
[project]
name = "cantocaptions-lang-en"
dependencies = ["cantocaptions-ai"]

[project.entry-points."cantocaptions_ai.languages"]
en = "cantocaptions_lang_en:EN"     # a LanguagePack, or a zero-argument function returning one
```

After `pip install cantocaptions-lang-en`, `--language en` uses it. Keep the module that defines the pack
light, since importing it is part of reading the registry: no torch, and heavy data (pronunciation tables,
OpenCC) behind factories, as `char_readings` and `builtin_steps` already are. A pack that fails to load is
skipped with a warning naming its distribution. One with the same code as a built-in pack replaces it, and
a pack registered in code with `register_language_pack` wins over both.

## Testing a pack

1. **Unit.** `get_language_pack("xx")` returns your pack, `resolve(None)` gives the expected model, and
   a `SubtitleCleaner` built from your `CleaningSpec` cleans a few representative lines.
2. **End to end, no models.** `tests/test_pipeline_e2e.py` runs the whole pipeline on synthetic audio
   that encodes its own transcript (`tests/_pipeline_fakes.py`). Copy
   `test_a_registered_pack_adds_a_fully_supported_language`: register the pack with `monkeypatch`, script
   a few utterances in the language, and assert on the cues. This checks your script, punctuation,
   layout and cleaning together in seconds.
3. **Real audio.** With transcribed audio and reference subtitles for the language,
   `scripts/build_eval_episodes.py` and `scripts/score_subtitles.py` measure CER, cue coverage and timing.
