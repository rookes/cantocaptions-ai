"""Language packs: everything that makes the pipeline speak one language.

A :class:`LanguagePack` gathers what used to be spread across the pipeline -- how the
language is written (script, punctuation), which models to use by default, how each ASR
model writes it (normalization, particle spot-checks, cue markers, cleaning manifest), its
cleaning rules, its audio-track tags, its LLM correction prompts and ensemble model. The
stages take the pieces they need from :meth:`LanguagePack.resolve`, and a new language is
one new pack (see ``docs/adding-a-language.md``).

Only the parts a language actually has are filled in. A language with no registered pack
still gets a :func:`generic_pack` -- its script and punctuation from the language code, an
align model from ``align_defaults`` -- which is enough to run the pipeline raw (no cleaning,
a model the user names).

This package imports nothing heavy: no torch, no OpenCC, no pycantonese. Packs defer those
to the functions that need them, so reading the registry stays cheap.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

from cantocaptions_ai.languages.align_defaults import default_align_model
from cantocaptions_ai.text_profiles import (
    DEFAULT_CLEANING,
    DEFAULT_NORMALIZATION,
    DEFAULT_SEGMENTATION,
    CleaningConfig,
    PunctuationConfig,
    ScriptConfig,
    SegmentationConfig,
    SpotCheck,
    TextNormalization,
    punctuation_for_script,
    script_for_language,
)


@dataclass(frozen=True)
class ModelConventions:
    """How one ASR model writes one language, where that differs from the pack's defaults.

    ``normalization`` is applied to the model's raw output (e.g. OpenCC for a model that
    writes Simplified characters); ``spotchecks`` let alignment swap interchangeable
    characters for the one the audio supports; ``segmentation`` names the clause-leading
    markers cue assembly rejoins; ``cleaning_manifest`` picks how much cleaning the output
    needs. ``punctuation``/``script`` override the pack's for a model that writes differently.
    Every field defaults to "nothing special", which is what an unlisted model gets.
    """
    normalization: TextNormalization = DEFAULT_NORMALIZATION
    spotchecks: Mapping[str, SpotCheck] = field(default_factory=dict)
    segmentation: SegmentationConfig = DEFAULT_SEGMENTATION
    cleaning_manifest: Optional[str] = None
    punctuation: Optional[PunctuationConfig] = None
    script: Optional[ScriptConfig] = None


def _no_builtins() -> Mapping[str, Callable[[str], str]]:
    return {}


@dataclass(frozen=True)
class CleaningSpec:
    """A language's cleaning rules (see ``cleaning/engine.py``).

    ``rules_dir`` holds the manifests and rule files; ``manifest`` is the one a model with
    no convention of its own gets. ``builtin_steps`` is a zero-argument loader for the coded
    steps a manifest may name -- deferred because they tend to pull in the language's NLP
    libraries. ``noise_tokens`` are lines dropped as pure interjection.
    """
    rules_dir: Path
    manifest: str = DEFAULT_CLEANING.manifest
    builtin_steps: Callable[[], Mapping[str, Callable[[str], str]]] = _no_builtins
    noise_tokens: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrectionPrompts:
    """The system prompts for one language's LLM correction passes (see llm_correction)."""
    particles: str              # pass A: per-segment particle / typo fix against the ensemble
    names: str                  # pass B: whole-document proper-noun consistency
    reference: str              # reference subtitle: homophone fixes only
    reference_semantic: str     # reference subtitle: also restore missing key words


@dataclass(frozen=True)
class RunProfile:
    """Everything the stages need for one run: a language pack resolved for one ASR model.

    The attribute names are the ones ``transcribe.py`` has always read off the model
    profile (``punctuation``, ``script``, ``spotchecks``, ``segmentation``,
    ``cleaning.manifest``), plus the resolved model itself.
    """
    language: str
    model: Optional[str]
    hf_id: Optional[str]
    normalization: TextNormalization
    punctuation: PunctuationConfig
    script: ScriptConfig
    spotchecks: Mapping[str, SpotCheck]
    segmentation: SegmentationConfig
    cleaning: CleaningConfig


@dataclass(frozen=True)
class LanguagePack:
    """One language, as the pipeline needs to know it. See the module docstring."""
    code: str
    script: ScriptConfig
    punctuation: PunctuationConfig
    # The ASR and align models a run uses when the config names none. No default ASR
    # model means the user must pick one.
    default_model: Optional[str] = None
    default_align_model: Optional[str] = None
    conventions: Mapping[str, ModelConventions] = field(default_factory=dict)
    cleaning: Optional[CleaningSpec] = None
    correction_prompts: Optional[CorrectionPrompts] = None
    # (hub repo, CTranslate2 subfolder) of the faster-whisper second opinion.
    ensemble_model: Optional[Tuple[str, str]] = None
    # streams -> index of the audio track to use; None matches the language code
    # (utils.audio.select_track).
    track_selector: Optional[Callable[[List[dict]], int]] = None

    @property
    def fully_supported(self) -> bool:
        """True if the pack can run end to end: a default model and cleaning rules."""
        return self.default_model is not None and self.cleaning is not None

    def model_name(self, model: Optional[str]) -> Optional[str]:
        """The ASR model a run uses: the configured one, else this language's default."""
        return model or self.default_model

    def conventions_for(self, model: Optional[str]) -> ModelConventions:
        name = self.model_name(model)
        return self.conventions.get(name, ModelConventions()) if name else ModelConventions()

    def resolve(self, model: Optional[str]) -> RunProfile:
        """This pack resolved for *model* (None: the pack's default model)."""
        name = self.model_name(model)
        conventions = self.conventions_for(name)
        manifest = conventions.cleaning_manifest or (
            self.cleaning.manifest if self.cleaning else DEFAULT_CLEANING.manifest)
        hf_id = None
        if name is not None:
            # Imported here: model_profiles reads this package to resolve a default model.
            from cantocaptions_ai.pipeline.model_profiles import get_model_profile
            hf_id = get_model_profile(name, self.code).hf_id
        return RunProfile(
            language=self.code,
            model=name,
            hf_id=hf_id,
            normalization=conventions.normalization,
            punctuation=conventions.punctuation or self.punctuation,
            script=conventions.script or self.script,
            spotchecks=conventions.spotchecks,
            segmentation=conventions.segmentation,
            cleaning=CleaningConfig(manifest=manifest),
        )

    def choose_track(self, streams: List[dict]) -> int:
        """The 0-based audio stream most likely to carry this language."""
        if self.track_selector is not None:
            return self.track_selector(streams)
        from cantocaptions_ai.utils.audio import select_track
        return select_track(streams, self.code)


def generic_pack(code: str) -> LanguagePack:
    """A pack for a language with none registered: enough to run the pipeline raw.

    Script and punctuation follow from the code (CJK for zh/ja/yue, space-separated
    otherwise); the align model is the built-in default for the language, if there is one.
    No default ASR model and no cleaning, so ``validate_config`` asks for ``--model`` and
    ``--no_clean_text``.
    """
    script = script_for_language(code)
    return LanguagePack(
        code=code,
        script=script,
        punctuation=punctuation_for_script(script),
        default_align_model=default_align_model(code),
    )
