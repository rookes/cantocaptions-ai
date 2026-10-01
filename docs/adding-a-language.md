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
    default_model="Qwen3-ASR",     # any MODEL_PROFILES key, or a hub id
    default_align_model="WAV2VEC2_ASR_BASE_960H",
)
```

Register it next to yue in `languages/__init__.py` (`_register_builtin_packs`), or from your own code
with `register_language_pack(EN)`. With only this, `--language en` needs no `--model`, but still needs
`--no_clean_text`: `fully_supported` means a default model **and** cleaning rules.

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
| `ensemble_model` | `(hub repo, CTranslate2 subfolder)` of a faster-whisper second opinion for `--ensemble_model`. |

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
