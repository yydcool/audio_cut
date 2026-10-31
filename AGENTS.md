# AGENTS.md

## Project

Template-based detection and lossless removal of repeating audio segments.
Given an input file and one or more template regions (default `0-26` s), the
tool finds every perceptually similar occurrence via spectrogram template
matching and removes them with `ffmpeg -c copy` (no re-encode; packets are
bit-preserved).

## Environment

- Python 3.14, `numpy`, `scipy` (already installed; see `requirements.txt` for fresh machines)
- `ffmpeg` / `ffprobe` on PATH (checks fail fast otherwise)
- No other dependencies. Do not add `librosa` etc. unless truly needed.

## Commands

```bash
python3 -m src --help                                       # CLI reference
python3 -m compileall -q src                                # syntax check

# detection only, with anchor verification (fast on re-runs via cache)
python3 -m src --input audio.m4a --template 0-26 --verify 07:00-07:26 --dry-run

# two patterns in one pass (--template is repeatable) -> matches labeled per template
python3 -m src --input audio.m4a --template 0-26 --template 06:50-06:59

# real lossless cut -> audio.cleaned.m4a + audio.cleaned.m4a.report/
python3 -m src --input audio.m4a --template 0-26 --verify 07:00-07:26

# batch: every audio file in download/ -> download/cleaned/ (+ reports/ + batch_summary.json)
python3 batch_process.py --input-dir download --template 0-26 --template 06:50-06:59

# batch with one reference file as template source for every file
# (regions default to 0-26 + 06:50-06:59 of the reference, override with --template)
python3 batch_process.py --input-dir download --template-source audio.m4a --jobs 4
```

Exit codes: `0` ok, `1` fatal error, `2` `--verify` anchor failed, `3` removal budget exceeded, `4` fewer occurrences than `--min-matches`.

## Architecture

Data flow: **decode -> features -> normalise -> NCC similarity -> peaks -> segments -> cut/report**

| Module | Responsibility |
| --- | --- |
| `src/__main__.py` | CLI parsing, orchestration, guard rails. No DSP logic. |
| `src/config.py` | All dataclass configs + shared `AudioCutError`. |
| `src/audio_io.py` | ffprobe duration; streaming ffmpeg decode to mono float32 chunks. |
| `src/features.py` | Streaming log-mel extraction, mel filterbank, global z-norm, feature cache. |
| `src/matcher.py` | Matcher interface, NCC matcher, peak picking, registries. |
| `src/segments.py` | `TimeSegment` algebra: pad, merge, drop/keep selection, budget checks. |
| `src/cutter.py` | Per-keep `ffmpeg -c copy` segments + concat demuxer. |
| `src/report.py` | `matches.json`, `scores.csv` (`scores_N.csv` per template when multiple), console formatting. |
| `batch_process.py` | Batch wrapper: runs the CLI in-process per file, `--template-source` reference file, `--jobs` parallel files, skip/overwrite, summary JSON. |

Rules:

- `__main__.py` must stay thin; algorithms live in their module.
- Cross-module errors: raise `AudioCutError` (printed without traceback in `main`).
- Frames are globally aligned: frame `f` covers samples `[f*hop, f*hop+n_fft)`,
  independent of decode chunking. Preserve this in new feature code.

## Extension points

- New feature type: subclass `FrameFeatureExtractor`, register with
  `@register_extractor("name")` in `features.py`, select via `--feature`.
- New matcher: subclass `Matcher`, register with `@register_matcher("name")`
  in `matcher.py`, select via `--matcher`.
- Nothing else (CLI, segments, cutter, reports) should need changes.

## Verification workflow

1. `python3 -m compileall -q src`
2. Synthetic smoke test: build a signal with a known repeated burst, run
   `StreamFeaturizer -> FeatureStats -> normalize_features -> matcher.similarity
   -> pick_peaks`, assert both occurrences are found. Keep such tests in `/tmp`,
   not the repo.
3. `--dry-run --verify ANCHOR` on the real file; confirm scores and intervals
   look right in `.report/matches.json` / `scores*.csv`.
4. After cutting, prove losslessness at **packet level** (not PCM):
   all kept packets in the output must be a CRC32-identical subsequence of the
   source (`ffprobe -show_entries packet=pts_time,size,data_hash -show_data_hash crc32`).

## Known gotchas

- **Never judge quality by comparing PCM from different decode paths.**
  ffmpeg's HE-AAC/AAC decoder output is path-dependent: decoding the *same file*
  with `-ss` vs continuously gives ~0.99 correlation. Compare packets instead.
- `scipy.signal.find_peaks` cannot see index 0; the template's own occurrence is
  kept via `pick_peaks(..., ensure_positions=...)` in `__main__.py`.
- Cut boundaries snap to the AAC frame grid (~23 ms at 44.1 kHz); removed
  durations deviate ~±50 ms from the requested interval. Expected, not a bug.
- Feature cache (`.audio_cut_cache/`, keyed by file stat + feature config) is
  invalidated automatically when the input changes; use `--no-cache` after
  changing feature code during development.
- Defaults encode user choices for this task: `drop_all=True` (remove even the
  original), auto-cut after detection, `--verify`/`--max-removed-fraction`
  guards abort before writing output unless `--force`.

## Style

- Module-level docstrings; no inline comments unless the why is non-obvious.
- Dataclasses for config/state (`config.py`), relative imports (`from .x import y`),
  type hints everywhere.
