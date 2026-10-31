# AGENTS.md

See `README.md` for usage, CLI reference, outputs, architecture map, and the full
verification workflow.

## Environment

- Python 3.14 with `numpy` and `scipy` installed; `ffmpeg` / `ffprobe` on PATH.
- Do not add dependencies (`librosa`, etc.) unless truly needed.

## Code rules

- `__main__.py` stays thin; algorithms live in their module.
- Cross-module errors raise `AudioCutError` (printed without traceback in `main`).
- Frames are globally aligned: frame `f` covers samples `[f*hop, f*hop+n_fft)`,
  independent of decode chunking. Preserve this in new feature code.
- Module-level docstrings; no inline comments unless the why is non-obvious.
- Dataclasses for config/state (`config.py`), relative imports, type hints everywhere.

## Gotchas

- `scipy.signal.find_peaks` cannot see index 0; the template's own occurrence is
  kept via `pick_peaks(..., ensure_positions=...)` in `__main__.py`.
- Defaults are deliberate: `drop_all=True`, auto-cut after detection, and
  `--verify` / `--max-removed-fraction` guards abort before writing unless `--force`.
- Keep ad-hoc tests in `/tmp`, not in the repo.

## Before finishing

- Run `python3 -m compileall -q src`.
- For detection or cutting changes, follow the verification steps in `README.md`
  (synthetic smoke test, `--dry-run --verify`, packet-level losslessness).
