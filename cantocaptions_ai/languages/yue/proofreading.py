"""Proofreading standards for Cantonese (``pipeline/proofread``).

One standard ships: CantoCaptions, the written-Cantonese house style the pack's cleaning
rules and default ASR model already follow. Another standard is one more
``ProofreadStandard`` here (and a conventions file beside it); a user can also keep this one
and swap only the conventions with ``--proofread_conventions``.

The examples in ``error_examples`` are deliberately generic sound-alikes, not taken from any
evaluation material: a prompt that quotes a test set's answers measures nothing.
"""
from cantocaptions_ai.languages.base import ProofreadStandard
from cantocaptions_ai.languages.yue.paths import PROOFREAD_DIR

CANTOCAPTIONS = ProofreadStandard(
    name="cantocaptions",
    language_name="written Cantonese (粵文)",
    description=(
        "The subtitles follow the CantoCaptions standard: written Cantonese for language "
        "learners, transcribing what was actually said, with each sentence-final particle "
        "written as the one character that matches its syllable and tone."
    ),
    error_examples="""\
- **Cantonese sound-alikes** (早神→早晨, 寄得→記得, 公員→公園) are the most valuable fixes.
- **Punctuation that changes the reading** — a question with no `？`, or an interrupted or
  trailing line with no `…`. Punctuation style (full-width forms, commas) is handled
  separately; leave it alone.
- **Spelling conventions** — a variant or spelling the conventions replace (俾→畀,
  咁樣→噉樣, 宜家→而家).""",
    conventions=PROOFREAD_DIR / "cantocaptions.md",
    # Standard Written Chinese characters colloquial Cantonese writes another way (的/嘅,
    # 這/呢, 那/嗰, 甚麼/乜嘢, 沒/冇, 很/好, 也/都, 是/係, 他們/佢哋, 吃/食, 看/睇, 說/講,
    # 既…又/唔止…又). The model borrows them from a Standard Chinese reference subtitle
    # while fixing something else in the same line. Each also occurs inside ordinary
    # Cantonese words (但是, 也許, 看法), so only characters an edit *introduces* count.
    # Measured over a few hundred labelled edits it fired rarely and only on bad ones.
    foreign_register=tuple("的這那甚麼沒很也是他她們吃看說誰哪怎既"),
    foreign_register_name="Standard Written Chinese",
    name_example="陳大文、大文、阿文",
    # Every particle the conventions file lists (its particle grid and compounds), less 下
    # and 罷, which end clauses as ordinary words too. Measured over whole-episode runs on
    # both the pipeline's output and human-checked text, the model's particle changes were
    # wrong about five times as often as right, most often on the rarer contractions (𠿪,
    # 𠾵, 𡁜). Told instead to change only common particles where the reference clearly marks
    # the function (了 -> 喇, 吧 -> 啦), it made no particle changes at all -- the same as
    # forbidding them, for a longer prompt -- and the one such change seen earlier was
    # wrong. Tone is what tells them apart, and the model cannot hear tone.
    final_particles=tuple("吖嗄啊呀咓𠻺𡅅噃𠺢𠿪㗎嘎㗇噶𠺝𠸏嘅啩吓嚱嗬啦𠸎喇嗱嚹嘞呢咧哩囖咯囉嚛嚕"
                          "嘛嗎咩𠻹哇喎啝𡁜吒咋喳𠾵咤啫唧"),
    final_particle_rule="""- **Sentence-final particles.** Leave every sentence-final particle (啊, 吖, 啦, 喇, 㗎, 咋,
  喎, 𠿪 …) exactly as the draft has it: do not swap, add or remove one, even where another
  reads more naturally. Which particle was said is a matter of syllable and tone, which you
  cannot hear, and the reference cannot tell you. The particle notes in the conventions are
  there so you can *read* the draft, not to correct it. If a particle looks plainly
  impossible, flag it. A missing `？` is still an error.""",
    final_particle_errors="""- **Wrong sentence-final particle** — a particle with the wrong syllable or tone for what
  the speaker is doing (喇 for a change of state against 啦 for urging; 吖 against 啊; 囖
  against 喇). The conventions map each syllable and tone to one character; use the particle
  notes to decide which one the context calls for.""",
)
