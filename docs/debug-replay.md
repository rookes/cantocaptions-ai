# Debug / replay checkpoints

What `--debug_dir` writes and what `--load_debug_dir` reloads.


`--debug_dir DIR` saves intermediate outputs (WAV segments + JSON) after each stage.  
`--load_debug_dir DIR` loads saved stages, skipping recomputation — useful for iterating on later stages without re-running expensive models.

Stage checkpoints: `vad`, `isolation`, `transcription`, `ensemble`, `llm_correction`, `diarization`, `realign`, `proofread`.

Each run's log (`{output_dir}/logs/`, or `--log_file`) is the place to read what a replay did: it
has every DEBUG line, so every checkpoint loaded or written, and why a stale one was recomputed.
The console only says `↺ Stage: loaded from DIR`.

`realign/` holds two files. `suspect.srt` is a *snapshot*, not a checkpoint: only the cues
carrying a reason, on their final timings, prefixed with that reason, for watching against the
film. `result.json` holds the coarse line placements (start, end, per-line CTC score, reason,
text) — the expensive half. Everything after it (regrouping, forced alignment, cleaning) is cheap or
wants to re-run anyway. It also doubles as the diagnostic dump: the per-line score is the only
view into how well the transcript matched the recording.

A file whose `vocal_isolation` checkpoint is cached skips the VAD stage entirely on replay (no model load, no reading the VAD WAVs back): the isolation checkpoint carries the same segment boundaries plus the audio ASR actually consumes, and stage 2 replaces `vad_segments` wholesale. The decision is per file, so a mixed `--input_dir` still runs VAD for the files that need it.

`diarization/` holds one loadable checkpoint (`result.json`: scope, speaker labels, exclusive turns, overlapping turns, the per-VAD-segment `segment_speakers` breakdown under segment scope, optional embeddings) plus two snapshots that are *not* reloaded — `assignments.json` and `{stem}.srt`, both showing the per-subsegment speaker, confidence and `!MULTI` flag. Assignment is cheap and must re-run so threshold changes apply on replay.

`notes/notes.srt` is a snapshot, not a checkpoint: every cue carrying a `notes` entry, on its
final timings, prefixed with them. Written on any run with a debug dir — nothing about it is
specific to `--realign` — and skipped entirely when nothing annotated anything.

`proofread/` holds one request/answer pair per chunk (the loadable checkpoint, keyed by request
hash) plus snapshots: `changes.srt` (each edited cue, new text over `[was]` old text),
`flags.srt` (lines the model doubted, and edits refused for confidence or register) and
`summary.json` (counts, cost, names).

`pre_cleaning/` (SRT + JSON snapshot of the merged segments before text cleaning) is written under the same tree but is *not* a loadable checkpoint — text cleaning always re-runs, so TOML rule edits take effect on `--load_debug_dir` replays.

