# Your configuration

This directory holds **your** settings for runs from this checkout. Git ignores what you
put here, except this file.

* `user.cfg`: your own defaults, layered over the shipped ones. Same format as
  [`cantocaptions_ai/presets/default.cfg`](../cantocaptions_ai/presets/default.cfg),
  holding only the keys you want to change, in a `[pipeline]` block or grouped by
  section (`[vad]`, `[alignment]`, ...). For example:

  ```ini
  [inference]
  batch_size = 24

  [vocal_isolation]
  vocal_isolation_method = mbroformer
  ```

* `NAME.cfg`: a preset for `--cfg NAME`. A file here named like a shipped preset
  (`default`, `cpu`, `fast_test`) replaces it.

Outside a checkout (a pip install), the same files go in `~/.config/cantocaptions-ai/`,
or wherever `$CANTOCAPTIONS_CONFIG_DIR` points.
