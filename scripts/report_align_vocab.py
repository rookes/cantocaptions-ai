"""Report the characters an align model has no token for, and what it would substitute.

The input to curating a substitution list. Reads a transcript (or an SRT) exactly the way
--realign does, asks the align model's own tokenizer which characters it holds, and runs
pipeline/align_vocab.py's resolver over the rest -- so what this prints is what a run would
actually do, not an approximation of it.

    uv run python scripts/report_align_vocab.py transcript.txt -o missing.md
    uv run python scripts/report_align_vocab.py transcript.txt --level near --toml

``--toml`` emits a ``[substitutions]`` table of every automatic choice, ready to hand-edit
and pass back as ``--align_substitutions``: keep the good ones, correct the bad ones, and set
a character to ``""`` to leave it alone.

Loads the tokenizer only, never the model weights, so it needs no GPU and runs in seconds.
"""
import argparse
import sys
import unicodedata
from collections import Counter, defaultdict

DEFAULT_MODEL = "alvanlii/wav2vec2-BERT-cantonese"


def build_dictionary(model_name: str, model_dir=None) -> dict:
    """The align model's vocabulary, keyed the way load_align_model keys it."""
    from transformers import Wav2Vec2BertProcessor, Wav2Vec2Processor

    cls = Wav2Vec2BertProcessor if "wav2vec2-BERT" in model_name else Wav2Vec2Processor
    processor = cls.from_pretrained(model_name, cache_dir=model_dir)
    return {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}


def bucket(char: str) -> str:
    if unicodedata.category(char) == "Nd":
        return "digit"
    if char.isascii() and char.isalpha():
        return "latin"
    if unicodedata.category(char)[0] in "PSZ":
        return "punctuation"
    return "letter" if unicodedata.category(char).startswith("L") else "other"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", help="line-delimited transcript, or an SRT/VTT")
    parser.add_argument("--align_model", default=DEFAULT_MODEL)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--level", default="near", choices=["off", "variant", "homophone", "near"],
                        help="resolver level to report; 'near' shows every tier (default)")
    parser.add_argument("--substitutions", default=None,
                        help="an existing override TOML to apply before reporting")
    parser.add_argument("-o", "--out", default=None, help="write Markdown here instead of stdout")
    parser.add_argument("--toml", action="store_true",
                        help="emit a [substitutions] table instead of the report")
    args = parser.parse_args(argv)

    from cantocaptions_ai.pipeline.align_vocab import (
        KIND_HELP, LEVELS, VocabRepair, _repairable, load_substitution_overrides, reading_of,
    )
    from cantocaptions_ai.pipeline.realign import REALIGN_PUNCTUATION, load_transcript_lines

    dictionary = build_dictionary(args.align_model, args.model_dir)
    known = set(dictionary)
    lines = load_transcript_lines(args.transcript)
    overrides = load_substitution_overrides(args.substitutions) if args.substitutions else None
    # A fresh dictionary per level so the tiers are reported independently: augment() mutates
    # the dictionary, and a character resolved at 'variant' is invisible to 'homophone'.
    repair = VocabRepair(dict(dictionary), args.level, overrides)

    counts: "Counter[str]" = Counter()
    line_hits = defaultdict(set)
    examples = {}
    chars = 0
    for line in lines:
        chars += len(line.text)
        for char in line.text:
            # .lower() because load_align_model lower-cases the vocabulary, so an uppercase
            # letter is alignable through its lowercase entry -- exactly what
            # _preprocess_segment does before its own membership test.
            if char.lower() in known or char in REALIGN_PUNCTUATION.split_chars:
                continue
            counts[char] += 1
            line_hits[char].add(line.index)
            examples.setdefault(char, line.text)

    resolved = {c: repair.resolve(c) for c in counts if _repairable(c)}

    if args.toml:
        out = ["# Substitutions for the align model's missing characters.",
               "# Edit freely, then pass with --align_substitutions.",
               '# An empty value means "leave this character alone".', "", "[substitutions]"]
        for char, sub in sorted(resolved.items(), key=lambda kv: -counts[kv[0]]):
            reading = reading_of(char) or "?"
            if sub is None:
                out.append(f'"{char}" = ""  # {reading}, x{counts[char]} — nothing found')
            else:
                out.append(f'"{char}" = "{sub.replacement}"  # {reading}, x{counts[char]}, '
                           f'{sub.kind}')
        text = "\n".join(out) + "\n"
    else:
        text = _markdown(args, lines, chars, counts, line_hits, examples, resolved, known,
                         KIND_HELP, LEVELS, dictionary, REALIGN_PUNCTUATION)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _markdown(args, lines, chars, counts, line_hits, examples, resolved, known, kind_help,
              levels, dictionary, punctuation) -> str:
    from cantocaptions_ai.pipeline.align_vocab import reading_of

    total = sum(counts.values())
    buckets = defaultdict(list)
    for char, n in counts.items():
        buckets[bucket(char)].append((n, char))

    # A line the aligner cannot see at all -- before and after substitution, so the report
    # says what the feature is worth on this transcript.
    def dead(after: bool) -> int:
        n = 0
        for line in lines:
            visible = any(
                c.lower() in known or (after and resolved.get(c) is not None)
                for c in line.text
                if c not in punctuation.split_chars
            )
            if not visible:
                n += 1
        return n

    out = []
    w = out.append
    w("# Characters missing from the align model vocabulary")
    w("")
    w(f"Transcript: `{args.transcript}`  ")
    w(f"Align model: `{args.align_model}` ({len(dictionary)} vocabulary entries)  ")
    w(f"Substitution level: `{args.level}`")
    w("")
    w(f"- {len(lines)} lines, {chars} characters")
    w(f"- **{total} characters ({total / max(chars, 1) * 100:.2f}%) have no token**, over "
      f"**{len(counts)} distinct characters**")
    w(f"- lines with nothing alignable in them: **{dead(False)}** before substitution, "
      f"**{dead(True)}** after")
    w("")

    letters = sorted(buckets.get("letter", []), reverse=True)
    if letters:
        by_kind = defaultdict(list)
        for n, char in letters:
            sub = resolved.get(char)
            by_kind[sub.kind if sub else "unresolved"].append((n, char, sub))
        w(f"## Letters — {len(letters)} distinct, {sum(n for n, _ in letters)} occurrences")
        w("")
        for kind in list(kind_help) + ["unresolved"]:
            rows = by_kind.get(kind)
            if not rows:
                continue
            note = kind_help.get(kind, "no substitute found; the character is dropped")
            w(f"### {kind} — {len(rows)} distinct, {sum(r[0] for r in rows)} occurrences")
            w("")
            w(f"*{note}*")
            w("")
            w("| char | jyutping | substitute | count | lines | example line |")
            w("|---|---|:-:|---:|---:|---|")
            for n, char, sub in rows:
                example = examples[char].replace("|", "\\|")
                w(f"| {char} | {reading_of(char) or ''} | {sub.replacement if sub else '—'} "
                  f"| {n} | {len(line_hits[char])} | {example} |")
            w("")

    for key, title, note in [
        ("digit", "Digits", "out of scope: the reading may be Cantonese, English or "
                            "digit-by-digit and nothing in the text says which"),
        ("latin", "Latin letters", "out of scope"),
        ("punctuation", "Punctuation and symbols",
         "out of scope: what the aligner wants is already mapped to blank, and a cue holding "
         "nothing else is noise"),
        ("other", "Other", ""),
    ]:
        rows = sorted(buckets.get(key, []), reverse=True)
        if not rows:
            continue
        w(f"## {title} — {len(rows)} distinct, {sum(n for n, _ in rows)} occurrences")
        if note:
            w("")
            w(f"*{note}*")
        w("")
        w("| char | count | lines | example line |")
        w("|---|---:|---:|---|")
        for n, char in rows:
            example = examples[char].replace("|", "\\|")
            w(f"| {char} | {n} | {len(line_hits[char])} | {example} |")
        w("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
