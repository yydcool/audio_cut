# audio_cut

Template-based detection and **lossless** removal of repeating audio segments.

Given an audio file and one or more template regions (by default `0-26` seconds),
`audio_cut` finds every perceptually similar occurrence with spectrogram template
matching, then removes them using `ffmpeg -c copy`. Nothing is re-encoded: the kept
audio packets are preserved bit-for-bit.

Typical use cases: stripping a recurring intro/jingle/outro from podcast or
audiobook episodes, or de-duplicating a segment repeated throughout a recording.

## Highlights

- **Lossless output** - stream copy only (`ffmpeg -c copy`), no PCM round-trip.
- **Robust matching** - normalised cross-correlation (NCC) over log-mel features,
  tolerant of level and codec differences.
- **Multiple patterns in one pass** - `--template` is repeatable; matches are
  labelled per template.
- **Safety guards** - anchor verification, minimum match count, and a removal
  budget stop a bad cut before anything is written.
- **Audit trail** - every run writes `matches.json`, similarity scores, and the
  exact kept/removed intervals.
- **Batch mode** - process a whole directory, optionally reusing one reference
  file's patterns across all files, with parallel workers and a summary report.
- **Feature cache** - re-runs skip decoding/featurization; invalidated
  automatically when the file or analysis settings change.

## Requirements

- Python 3.10+ (developed on 3.14)
- `numpy` and `scipy` (see `requirements.txt`)
- `ffmpeg` and `ffprobe` on `PATH`

## Installation

```bash
git clone <repo-url> audio_cut
cd audio_cut
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ffmpeg -version   # must succeed
```

No other dependencies. The tool is run from the repository root:

```bash
python3 -m src --help
```

## Quick start

Detect only, without writing audio:

```bash
python3 -m src --input audio.m4a --template 0-26 --dry-run
```

Detect and cut:

```bash
python3 -m src --input audio.m4a --template 0-26
```

This writes `audio.cleaned.m4a` next to the input and an audit directory at
`audio.cleaned.m4a.report/`.

Verify a known occurrence while cutting:

```bash
python3 -m src --input audio.m4a --template 0-26 --verify 07:00-07:26
```

If the anchor at `07:00` is not found within the tolerance, the run exits with
code `2` before producing output.

### Removing several patterns

`--template` can be given more than once; all matches are combined in one pass:

```bash
python3 -m src --input audio.m4a --template 0-26 --template 06:50-06:59
```

Regions use `SS`, `MM:SS`, or `HH:MM:SS` (fractional seconds allowed), and must
satisfy `START < END`. By default every occurrence is removed, **including the
original template region**. Use `--keep-first` to keep the first occurrence and
remove only later repeats.

### Batch processing

Process every audio file directly inside a directory:

```bash
python3 batch_process.py --input-dir download --template 0-26 --template 06:50-06:59
```

Outputs go to `download/cleaned/`, per-file reports to
`download/cleaned/reports/<name>.report/`, and a summary to
`download/cleaned/reports/batch_summary.json`. Existing outputs are skipped
unless `--overwrite` is passed.

Use one reference file as the template source for every input (useful when the
pattern sits at different offsets in each file):

```bash
python3 batch_process.py --input-dir download --template-source audio.m4a --jobs 4
```

With `--template-source` and no explicit `--template`, the regions default to
`0-26` and `06:50-06:59` of the reference file. `--jobs N` processes up to `N`
files in parallel threads; decode runs in ffmpeg subprocesses and numpy releases
the GIL, so throughput scales with core count.

## Outputs

For a single-file run:

| Path | Contents |
| --- | --- |
| `<stem>.cleaned<suffix>` | Cut audio, produced without re-encoding |
| `<output>.report/matches.json` | Full audit: config, matches, candidates, dropped/kept intervals, verification |
| `<output>.report/scores.csv` | Similarity over time (`scores_N.csv` per template when multiple) |
| `<output>.work/` | Temporary segments during cutting; removed afterwards |

`matches.json` includes the input/output paths, feature and match configuration,
per-template summary, every match with its score, lower-threshold candidate
peaks, the final `dropped` and `keeps` intervals, removal time/fraction, and
whether the removal budget was respected. This makes a dry run fully
reproducible without touching the audio.

## How it works

```
decode -> log-mel features -> global normalisation -> NCC similarity
       -> peak picking -> merge/pad segments -> lossless cut -> report
```

1. **Decode** - ffprobe reads the duration; ffmpeg streams mono float32 audio at
   the analysis sample rate in chunks (default 600 s), so long files never need
   to fit in memory.
2. **Features** - a streaming log-mel spectrogram is computed on a globally
   aligned frame grid (frame `f` covers samples `[f*hop, f*hop+n_fft)`), making
   the result independent of chunk boundaries.
3. **Normalise** - per-band mean/std are accumulated over the whole file and
   applied globally.
4. **Match** - each template's normalised features are slid over the file with
   normalised cross-correlation. Peaks above `--threshold` and spaced at least
   `--min-peak-distance` apart are accepted; the template's own position is
   always considered.
5. **Select** - occurrences are padded by `--pad`, merged if they touch, and
   either all dropped or all but the first (`--keep-first`).
6. **Cut** - each kept interval is extracted with `ffmpeg -ss ... -c copy`,
   then concatenated with the concat demuxer. If nothing is removed, the input
   is copied verbatim.
7. **Report** - audit artifacts are written before any audio is cut.

## CLI reference

Run `python3 -m src --help` for the authoritative list.

### Input and templates

| Option | Default | Description |
| --- | --- | --- |
| `--input FILE` | required | Source audio file |
| `--template START-END` | `0-26` | Template region; repeatable |
| `--template-source FILE` | none | Read template regions from this reference file instead of the input |
| `--output FILE` | `<stem>.cleaned<suffix>` | Output audio path |

### Detection

| Option | Default | Description |
| --- | --- | --- |
| `--threshold F` | `0.85` | Minimum NCC similarity for a match |
| `--min-peak-distance S` | `0.9 x template duration` | Minimum spacing between matches |
| `--feature NAME` | `logmel` | Feature extractor |
| `--matcher NAME` | `ncc` | Matcher backend |
| `--sample-rate HZ` | `16000` | Analysis sample rate |
| `--n-fft N` | `2048` | FFT size |
| `--hop S` | `0.1` | Analysis hop, seconds |
| `--n-mels N` | `64` | Mel bands |
| `--fmin HZ` / `--fmax HZ` | `30` / Nyquist | Mel frequency range |

### Cutting and outputs

| Option | Default | Description |
| --- | --- | --- |
| `--pad S` | `0.2` | Extra margin removed around each occurrence |
| `--keep-first` | off | Keep the first occurrence; remove only repeats |
| `--work-dir DIR` | `<output>.work` | Scratch directory for segments |
| `--report-dir DIR` | `<output>.report` | Audit artifact directory |
| `--keep-work` | off | Keep intermediate segment files |
| `--ffmpeg` / `--ffprobe` | `ffmpeg` / `ffprobe` | Binary names or paths |
| `--dry-run` | off | Detect and report only; write no audio |

### Safety and performance

| Option | Default | Description |
| --- | --- | --- |
| `--verify START-END` | none | Anchor that must be detected, or the cut aborts |
| `--verify-tolerance S` | `2.0` | Accepted start offset for the anchor |
| `--min-matches N` | `1` | Abort if fewer occurrences are detected |
| `--max-removed-fraction F` | `0.15` | Abort if more than this share would be removed |
| `--force` | off | Continue despite failed verify/budget/min-match guards |
| `--cache-dir DIR` | `.audio_cut_cache` | Feature cache location |
| `--no-cache` | off | Disable the feature cache |

`batch_process.py` accepts the relevant subset (`--template`, `--template-source`,
`--threshold`, `--pad`, `--min-occurrences`, `--max-removed-fraction`,
`--verify`, `--keep-first`, `--dry-run`, `--no-cache`) plus `--input-dir`,
`--output-dir`, `--overwrite`, and `--jobs`. Its `--min-occurrences` defaults to
`2`, so files without a real repeat are skipped rather than copied.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (including dry runs) |
| `1` | Fatal error (missing input, ffmpeg failure, bad arguments) |
| `2` | `--verify` anchor not found (aborted before cutting) |
| `3` | Removal budget `--max-removed-fraction` exceeded |
| `4` | Fewer occurrences than `--min-matches` |

`--force` overrides codes `2`, `3`, and `4`. Batch mode returns `1` if any file
failed and `130` if interrupted.

## Losslessness and boundaries

No audio samples are ever decoded and re-encoded for the output. Kept intervals
are copied as packets and concatenated, so quality is preserved exactly.

Because stream copy can only cut on the container's packet/frame grid, cut
boundaries snap to that grid (about 23 ms for 44.1 kHz AAC). Removed durations
therefore deviate by roughly +/-50 ms from the requested interval; this is
expected, not a bug.

To verify losslessness, compare **packets**, not decoded PCM:

```bash
ffprobe -v error -select_streams a:0 \
  -show_entries packet=pts_time,size,data_hash -show_data_hash crc32 \
  -of csv=p=0 audio.m4a > source_packets.csv

ffprobe -v error -select_streams a:0 \
  -show_entries packet=pts_time,size,data_hash -show_data_hash crc32 \
  -of csv=p=0 audio.cleaned.m4a > cleaned_packets.csv
```

Every kept packet in `cleaned_packets.csv` should be a CRC32-identical
subsequence of `source_packets.csv`.

Do **not** judge quality by comparing PCM decoded from different paths: ffmpeg's
HE-AAC/AAC decoder is path-dependent, and decoding the same file continuously
versus with `-ss` can differ by around 1% correlation while the packets are
identical.

## Feature cache

Features are cached in `.audio_cut_cache/`, keyed by file name, size, mtime, and
the feature configuration. The cache is invalidated automatically when the input
or settings change. During feature development use `--no-cache` to avoid stale
results.

## Project layout

| Module | Responsibility |
| --- | --- |
| `src/__main__.py` | CLI parsing, orchestration, guard rails; no DSP logic |
| `src/config.py` | Configuration dataclasses and shared `AudioCutError` |
| `src/audio_io.py` | ffprobe duration; streaming ffmpeg decode to mono float32 |
| `src/features.py` | Streaming log-mel extraction, mel filterbank, normalisation, cache |
| `src/matcher.py` | Matcher interface, NCC matcher, peak picking, registries |
| `src/segments.py` | Interval algebra: pad, merge, drop/keep selection, budget checks |
| `src/cutter.py` | Per-keep `ffmpeg -c copy` extraction and concat |
| `src/report.py` | `matches.json`, `scores*.csv`, console formatting |
| `batch_process.py` | Batch wrapper: reference templates, parallel files, summary JSON |

### Extending

- **New feature type** - subclass `FrameFeatureExtractor`, register it with
  `@register_extractor("name")` in `src/features.py`, then select it with
  `--feature name`.
- **New matcher** - subclass `Matcher`, register it with
  `@register_matcher("name")` in `src/matcher.py`, then select it with
  `--matcher name`.

The CLI, segment logic, cutter, and reports should not need changes for either.

## Development

```bash
python3 -m compileall -q src    # syntax check
python3 -m src --help           # CLI smoke check
```

Recommended verification loop when changing detection code:

1. Compile-check the package.
2. Build a synthetic signal with a known repeated burst in `/tmp` and assert
   that `StreamFeaturizer -> FeatureStats -> normalize_features -> matcher.similarity
   -> pick_peaks` finds both occurrences.
3. Run `--dry-run --verify ANCHOR` on a real file and inspect
   `.report/matches.json` and `scores*.csv`.
4. After a real cut, prove packet-level losslessness with the ffprobe command
   above.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| No matches found | Lower `--threshold` slightly (e.g. `0.8`), or confirm the template is correct |
| Too many false matches | Raise `--threshold`, lengthen the template, or raise `--min-peak-distance` |
| Anchor verification fails | Check the `--verify` range and tolerance; inspect `scores.csv` |
| Removal budget exceeded | Expected for heavy repetition; adjust `--max-removed-fraction` or use `--force` |
| Stale results after code changes | Re-run with `--no-cache` |
| `ffmpeg`/`ffprobe` not found | Install them or pass `--ffmpeg`/`--ffprobe` paths |
| Boundaries off by tens of ms | Inherent to lossless frame-grid cuts; see above |

## Design notes

- Matching operates on normalised log-mel correlation, which is more stable
  across codec and loudness differences than raw waveform comparison.
- Normalisation statistics span the whole file, so similarity scores are
  comparable across the timeline.
- Reports are written before cutting, so a guard failure still leaves a complete
  audit of what would have happened.
- Feature caching plus parallel batch workers make re-processing large
  collections practical.
