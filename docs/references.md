# Embedded reference subtitles

`cantocaptions_ai/references/` writes a media file's own subtitle tracks out as reference SRTs:
`--extract_references` before a run, `--references_only` with no run, and
`--reference_subtitle embedded` to extract them and use the best Chinese track as the run's
reference. It replaces the external sub-convert tool for OCR of image tracks.

## Flow

1. **Probe** (`probe.py`). ffprobe lists the subtitle streams. Each is `text` (subrip, ASS/SSA,
   mov_text, WebVTT), `bitmap` (PGS, DVB, VobSub) or `other` (teletext...), never extracted.
   `--reference_tracks` picks `chinese` (default), `all` or stream indices. A track's language
   tag and title give its file-name tag (`zh-Hant`, `zh-Hans`, `en`...) and its OCR language.
2. **Text tracks** (`text.py`): `ffmpeg -map 0:N -c:s srt`, read back with
   `utils/subtitles.read_subtitle_cues`, markup stripped, repeated ASS layers collapsed.
3. **Bitmap tracks** go to the OCR worker as one batch (`extract._run_worker`), so the model
   loads once.
4. **Manifest**: `NAME.references.json` records every stream, what became of it, and
   `best_chinese` (`extract.best_chinese`): a full track before a signs/forced one, text before
   OCR, Traditional before unknown before Simplified, then more cues.
5. **Pipeline step** (`step.py`): runs before VAD (VAD expansion and ASR context read the
   reference) and finishes before any speech model loads. Checkpointed under
   `{debug_dir}/{name}/references/` (see [debug-replay.md](debug-replay.md)).

The files sit on the media's own timeline (ffmpeg rebases both text and bitmap tracks to the
file's start). Under `--audio_start` an embedded reference is treated like any other
`--reference_subtitle`.

## Bitmap tracks

### Rendering: ffmpeg sub2video

Routing a subtitle stream into a filtergraph turns on ffmpeg's sub2video mode, which paints
each subtitle update onto a transparent canvas with the palette applied, for any bitmap codec:

```
ffmpeg -canvas_size WxH -i F -filter_complex "[0:N]null[out]" -map "[out]" \
       -fps_mode passthrough -frame_pts true -f image2 DIR/%d.png
```

- `-fps_mode passthrough` gives one frame per update, not one per video tick.
- `-frame_pts true` names each PNG by its time in microseconds, the only timing we get.
- `-canvas_size`: without it the canvas is the *video's* size and ffmpeg scales the subtitle to
  fit. A 1920x1040 encode of a 1920x1080 Blu-ray shrank every PGS caption by 4% (text rows
  887-965 became 854-929). The canvas is the format's own (1920x1080 for PGS and DVB, 720x576
  for VobSub) or the video's, whichever is bigger.

The technique is the one sub-convert's `docs/raster.md` describes for DVB and VobSub; no
sub-convert code is used. sub-convert is AGPL-3.0, and this repository is BSD-2-Clause.

Frames come in pairs: the last moment of one state and the first of the next (a caption at
90.799000, again at 91.924999, a blank at 91.925000). An event is a run of frames with the same
visible-text mask (alpha >= 128), so a fade, a PGS re-send or a heartbeat repeat stays one event.
Keys are computed on 8 threads (Pillow decodes without the GIL): 30.7 s became 5.5 s for 1114
full-HD frames.

### Cutting lines

Each event's picture is cut into lines (rows of text split at blank rows, with slivers under
40% of the tallest line merged back: the second stroke of 二, a dash). A line is cut again at
every blank at least one line-height wide: two speakers side by side
(`你想我現在放手是嗎？　　沒錯`) or `02:20後　等候訊號行動`. Read whole, the recogniser filled that
gap with a character that is not there (妳, ，). The pieces are joined with a space.

### Image preparation: fill only

Subtitle text is a fill inside an outline of the opposite tone. `raster._ink` keeps only the
fill, as dark ink on white, weighted by alpha. Which tone is the fill is read from the picture:
the outline is what touches transparency. Flattening outline and all onto grey (what sub-convert
does) gave the recogniser a thick ring around each stroke. With PP-OCRv5, on Ghost in the Shell
S01E01, it then read the outlined fullwidth comma as a middle dot (課長・現場 for 課長，現場), an
ellipsis as ".…", and digits in fullwidth form (1７分鐘). With PP-OCRv6 it drops those marks
instead. Against grey, fill-only changed 25 cues of GitS S01E01 and AoT S2E26:
- better in 23: the commas and ellipses kept (帕茲，波瑪, 那是…, 嗯…), 久保田 not 久保, 問 not 間,
  艾倫、米卡莎 not 艾倫·米卡莎;
- worse in 2: a space inside "9 課", and 带 for 帶.

### Recognition

`line` mode (the default) runs PaddleOCR's text recogniser alone on each piece. A subtitle line
is already the recogniser's input: one horizontal line of text. The full detect-then-recognise
pipeline (`detect` mode) boxes a lone 、 or the dots of an ellipsis as text of their own and
reads them as "0", "000", "D" or "C". It also splits one line at a wide gap.
Line mode against detect mode on GitS S01E01 (PP-OCRv5): 27 cues differ. In each one checked,
detect mode was the one wrong, with a junk character or a lost mark (就別再聽0別再看,
那麼1為什麼, 如果對方想要C一之瀨報告77).

A Latin piece wider than 12 times its height is re-read in halves, cut at a word space: on the
end-song lines of AoT S2E26 the recogniser dropped and merged letters across a long, low line
("Surely come in the morngone day sy ello"), and read each half exactly. Chinese is never cut.
Halving every wide piece changed 9 Chinese cues on GitS for the worse (PP-OCRv5) (管制 to 管製,
判斷 to 判断, 一之瀨 to —之瀨) and fixed none, so the re-read is limited to Latin text.

Batching 32 pieces against reading them one at a time (PP-OCRv5): 2 cues differ either way on
each of GitS and AoT, a wash, so batches of 32 are kept.

### Model: PP-OCRv6_medium_rec

PaddleOCR 3.4 served `ch` and `chinese_cht` with PP-OCRv5_server_rec; 3.7 adds PP-OCRv6. These
five tracks were run through the same pipeline with each model:

| track | type | cues | v6 vs v5: cues differing |
|---|---|---|---|
| Ghost in the Shell S01E01 | PGS, 1080p | 278 | 3 |
| Attack on Titan S2E26 | PGS, 1080p | 308 | 5 |
| Sailor Moon 01 (TVB) | PGS, 1080p | 292 | 25 |
| Triumph in the Skies E01 | VobSub, 576p | 579 | 22 |
| The Bund E01 | VobSub, 480p | 489 | 7 |

Checked against the pictures, v6 is right on the words that change the meaning. It reads 幹
where v5 wrote 乾 or 干 (11 cues), 奸 where v5 wrote 姦 (4), and also 兇 (v5 兒), 店舗 (v5
dropped 舗), 丁嬸 (v5 dropped 嬸), 決定, 注意, Samuel, Zoe and X小姐. It keeps "..." intact
where v5 lost dots.

v6's own errors:
- the enumeration comma 、 dropped in 7 cues (經理、機長 read as 經理機長);
- Peggy、Iris read as "Peggylris";
- a few characters in their Simplified form (盗, 毁, 带, 遗);
- Y read as 丫 once.

None of these changes what a line means, except the merged English name. v6 is also faster:
16 s against 27 s for AoT S2E26.

paddlex 3.7 requires `numpy<2.4`. The lock holds one numpy for every extra, so the base install
moved from numpy 2.4.6 to 2.3.5 (Python 3.11+; 3.10 stays on 2.2.6, see
`constraint-dependencies` in pyproject.toml). The whole test suite, golden end-to-end outputs
included, passes unchanged on 2.3.5.

### Against sub-convert

Start times are within 0.1 s of sub-convert's on every cue of all five tracks, and the cue
counts are equal. docCER of our text against sub-convert's (neither is ground truth):

| track | docCER vs sub-convert | time (ours / sub-convert) |
|---|---|---|
| GitS S01E01 | 4.65% | 34 s / 236 s* |
| AoT S2E26 | 0.80% | 36 s / 67 s |
| Sailor Moon 01 | 1.18% | 34 s / 63 s |
| Triumph in the Skies E01 | 2.27% | 26 s / 101 s |
| The Bund E01 | 0.03% | 25 s / 71 s |

\* sub-convert's first run, which may include a model download. Ours is the wall time of
`cantocaptions FILE --references_only`, from process start, model load (about 8 s) and the
ffmpeg render included.

Most of the GitS difference is sub-convert's own errors:
- junk lines from the detector ("000", "D", "7", "£5") on about 30 cues;
- the lines of one caption split and reversed ("沒錯⏎你想我現在放手是嗎？", "大臣？⏎沒事嗎");
- 明白 read as 明自 throughout.

On VobSub, sub-convert also reverses the two halves of a two-speaker line (我怎麼知道⏎你請我吃什麼).
We keep them in reading order on one line, joined by a space.

`scripts/eval_references.py HYP BASE ... --diff N` reproduces the comparisons.

## The worker process

`worker.py` runs OCR in a child process (`python -m cantocaptions_ai.references.worker`).
Paddle and torch each bundle CUDA, cuDNN and OpenMP runtimes, and on Windows the second one
loaded fails. Paddle's allocator also holds its VRAM until the process exits. The worker exits
before VAD starts, so the speech models get the whole card.

The worker also hides torch from `importlib.util.find_spec`. paddlex imports modelscope, whose
logger imports torch whenever `find_spec("torch")` finds it. With Paddle's CUDA already loaded,
that import fails on Windows (`WinError 127` loading `torch\lib\shm.dll`) and takes `import
paddleocr` with it. `sys.modules["torch"] = None` would hide it too, but SciPy's array-API
helpers then read that None as the module and crash.

`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` stops paddlex probing its model hosters on every start;
`--model_cache_only` also sets `HF_HUB_OFFLINE`. Models download once, to `~/.paddlex`.

## Dependencies

The `ocr` extra pulls paddlepaddle-gpu 3.3.1 from Paddle's own index: cu129 on Windows, whose
wheel bundles its CUDA DLLs, and **cu118 on Linux**. On Linux both cu129 and cu126 pin
`nvidia-*-cu12` versions that conflict with torch cu128's own pins (cublas 12.9.0.13 against
12.8.4.1, cuDNN 9.9 against 9.19), and uv cannot resolve them together. cu118's libraries are the
`-cu11` packages, a different name and soname, so they install beside torch's. CUDA 11.8 has no
kernels for Blackwell (sm_120) cards; on those, Linux OCR needs `--reference_ocr_device cpu`.
macOS gets the CPU `paddlepaddle`.
