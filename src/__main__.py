"""CLI orchestration: parse arguments, run detection, guard rails, cut."""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio_io import probe_duration, require_binary
from .config import (
    AudioConfig,
    AudioCutError,
    CutConfig,
    FeatureConfig,
    MatchConfig,
    RunConfig,
)
from .cutter import LosslessCutter
from .features import (
    build_extractor,
    load_or_build_features,
    normalize_features,
)
from .matcher import build_matcher, pick_peaks
from .report import format_timestamp, print_match_list, write_reports
from .segments import (
    TimeSegment,
    check_removal_budget,
    complement_segments,
    find_anchor_match,
    matches_to_segments,
    merge_segments,
    pad_segment,
    select_dropped,
)

PREFIX = "[audio_cut]"
_TIME_TOKEN = re.compile(r"^\s*([\d:.]+)\s*-\s*([\d:.]+)\s*$")


def parse_time(text: str) -> float:
    """Accept ``SS``, ``MM:SS``, ``HH:MM:SS`` (fractional seconds allowed)."""
    parts = text.strip().split(":")
    if not parts or len(parts) > 3:
        raise argparse.ArgumentTypeError(f"invalid time '{text}'")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid time '{text}'") from exc
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"negative time '{text}'")
    return seconds


def parse_range(text: str) -> tuple[float, float]:
    match = _TIME_TOKEN.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid range '{text}' (expected START-END)")
    start = parse_time(match.group(1))
    end = parse_time(match.group(2))
    if end <= start:
        raise argparse.ArgumentTypeError(f"range end must be after start: '{text}'")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Detect repeats of a template region with spectrogram template "
            "matching and remove them losslessly (ffmpeg -c copy)."
        ),
    )
    parser.add_argument("--input", required=True, help="source audio file (e.g. audio.m4a)")
    parser.add_argument(
        "--output",
        default=None,
        help="output file (default: <input stem>.cleaned<suffix> next to the input)",
    )
    parser.add_argument(
        "--template",
        type=parse_range,
        action="append",
        default=None,
        metavar="START-END",
        help=(
            "template region, repeatable for multiple patterns, e.g. "
            "--template 0-26 --template 06:50-06:59 (default: 0-26)"
        ),
    )
    parser.add_argument(
        "--template-source",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "read --template regions from this file instead of the input, to "
            "reuse one reference file's patterns across many files"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="minimum NCC similarity for a match (default: 0.85)",
    )
    parser.add_argument(
        "--min-peak-distance",
        type=float,
        default=None,
        metavar="SECONDS",
        help="minimum spacing between matches (default: 0.9x template duration)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.2,
        help="extra margin removed around each occurrence, seconds (default: 0.2)",
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        help="keep the first occurrence, remove only later repeats (default: drop all)",
    )
    parser.add_argument(
        "--max-removed-fraction",
        type=float,
        default=0.15,
        help="abort if more than this share of the file would be removed (default: 0.15)",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=1,
        help="abort before cutting when fewer occurrences were detected (default: 1)",
    )
    parser.add_argument(
        "--verify",
        type=parse_range,
        default=None,
        metavar="START-END",
        help="anchor range that must be detected, e.g. 08:51-09:17; aborts the cut if not found",
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=2.0,
        help="accepted start offset for --verify, seconds (default: 2.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="detect and report only; do not write the output file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue despite failed --verify or removal budget guard",
    )
    parser.add_argument("--feature", default="logmel", help="feature extractor (default: logmel)")
    parser.add_argument("--matcher", default="ncc", help="matcher backend (default: ncc)")
    parser.add_argument("--sample-rate", type=int, default=16000, help="analysis sample rate")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT size (default: 2048)")
    parser.add_argument(
        "--hop", type=float, default=0.1, help="analysis hop in seconds (default: 0.1)"
    )
    parser.add_argument("--n-mels", type=int, default=64, help="mel bands (default: 64)")
    parser.add_argument("--fmin", type=float, default=30.0, help="mel low edge Hz (default: 30)")
    parser.add_argument("--fmax", type=float, default=None, help="mel high edge Hz (default: Nyquist)")
    parser.add_argument(
        "--cache-dir",
        default=".audio_cut_cache",
        help="directory for cached features (default: .audio_cut_cache)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="disable the feature cache"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="scratch directory for cut segments (default: <output>.work)",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="directory for matches.json/scores.csv (default: <output>.report)",
    )
    parser.add_argument(
        "--keep-work", action="store_true", help="keep intermediate segment files"
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe binary")
    return parser


def _log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise AudioCutError(f"input file not found: {input_path}")
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.cleaned{input_path.suffix}")
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output_path.with_name(output_path.name + ".work")
    )
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else output_path.with_name(output_path.name + ".report")
    )
    return RunConfig(
        input_path=str(input_path),
        audio=AudioConfig(
            sample_rate=args.sample_rate,
        ),
        features=FeatureConfig(
            name=args.feature,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_seconds=args.hop,
            n_mels=args.n_mels,
            fmin=args.fmin,
            fmax=args.fmax,
        ),
        match=MatchConfig(
            templates=args.template or [(0.0, 26.0)],
            threshold=args.threshold,
            min_peak_distance_seconds=args.min_peak_distance,
            drop_all=not args.keep_first,
            pad_seconds=args.pad,
            max_removed_fraction=args.max_removed_fraction,
            min_matches=args.min_matches,
            verify=args.verify,
            verify_tolerance_seconds=args.verify_tolerance,
        ),
        cut=CutConfig(
            output_path=str(output_path),
            work_dir=str(work_dir),
            report_dir=str(report_dir),
            dry_run=args.dry_run,
            force=args.force,
            keep_segments=args.keep_work,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        ),
    )


def run(args: argparse.Namespace) -> int:
    started = time.time()
    config = _build_run_config(args)
    input_path = Path(config.input_path)
    output_path = Path(config.cut.output_path)
    match_config = config.match
    feature_config = config.features
    audio_config = config.audio

    require_binary(config.cut.ffmpeg)
    require_binary(config.cut.ffprobe)
    duration = probe_duration(input_path, ffprobe=config.cut.ffprobe)
    _log(f"input: {input_path} ({format_timestamp(duration)})")

    template_source_path: Path | None = None
    if args.template_source is not None:
        candidate = Path(args.template_source).expanduser()
        if not candidate.is_file():
            raise AudioCutError(f"template source not found: {candidate}")
        if candidate.resolve() != input_path.resolve():
            template_source_path = candidate

    template_file_duration = duration
    if template_source_path is not None:
        template_file_duration = probe_duration(
            template_source_path, ffprobe=config.cut.ffprobe
        )
        _log(
            f"template source: {template_source_path} "
            f"({format_timestamp(template_file_duration)})"
        )

    template_origin = "template source" if template_source_path is not None else "the file"
    for template_start, template_end in match_config.templates:
        if not 0.0 <= template_start < template_end <= template_file_duration:
            raise AudioCutError(
                f"template {template_start}-{template_end}s "
                f"outside {template_origin} (0-{template_file_duration:.3f}s)"
            )

    extractor = build_extractor(feature_config)
    _log(
        f"featurizing ({feature_config.name}, {extractor.sample_rate} Hz, "
        f"hop {extractor.hop_seconds:.3f}s, {extractor.n_features} bands)"
    )

    def on_chunk(index: int, samples: int, frames: int) -> None:
        _log(f"  chunk {index}: {samples / extractor.sample_rate / 60:.1f} min decoded")

    cache_dir = None if args.no_cache else args.cache_dir
    features, stats, from_cache = load_or_build_features(
        input_path,
        extractor,
        audio_config,
        cache_dir=cache_dir,
        ffmpeg=config.cut.ffmpeg,
        progress=on_chunk,
    )
    _log(
        f"features: {features.shape[0]} frames x {features.shape[1]} bands "
        f"({'cache' if from_cache else 'computed'})"
    )
    normalized = normalize_features(features, *stats.finalize())

    if template_source_path is None:
        template_features = normalized
    else:
        source_features, source_stats, source_from_cache = load_or_build_features(
            template_source_path,
            extractor,
            audio_config,
            cache_dir=cache_dir,
            ffmpeg=config.cut.ffmpeg,
            progress=on_chunk,
        )
        template_features = normalize_features(source_features, *source_stats.finalize())
        _log(
            f"template-source features: {source_features.shape[0]} frames x "
            f"{source_features.shape[1]} bands "
            f"({'cache' if source_from_cache else 'computed'})"
        )

    hop = extractor.hop_seconds
    matcher = build_matcher(args.matcher)
    audit_floor = min(match_config.threshold, max(0.5, match_config.threshold - 0.3))
    matches: list[TimeSegment] = []
    audit_segments: list[TimeSegment] = []
    similarity_curves: list[tuple[str, np.ndarray]] = []
    per_template: list[dict] = []

    for template_start, template_end in match_config.templates:
        first_frame = int(round(template_start / hop))
        last_frame = int(round(template_end / hop))
        if last_frame > template_features.shape[0]:
            raise AudioCutError(
                f"template {format_timestamp(template_start)}-"
                f"{format_timestamp(template_end)} extends beyond the "
                f"{template_origin} audio available after featurization"
            )
        template = template_features[first_frame:last_frame]
        template_duration = (last_frame - first_frame) * hop
        label = f"{format_timestamp(template_start)}-{format_timestamp(template_end)}"
        _log(f"template {label} ({template_duration:.3f}s)")

        similarity = matcher.similarity(template, normalized)
        if similarity.size == 0:
            raise AudioCutError(
                f"similarity curve is empty for template {label}; "
                "template longer than the file?"
            )
        _log(
            f"  similarity: mean {float(similarity.mean()):.3f}, "
            f"max {float(similarity.max()):.3f}"
        )

        min_distance = match_config.min_peak_distance_seconds
        if min_distance is None:
            min_distance = 0.9 * template_duration
        min_distance_frames = int(round(min_distance / hop))
        ensure_positions = [0, similarity.size - 1]
        if template_source_path is None:
            ensure_positions.append(first_frame)
        peaks = pick_peaks(
            similarity,
            match_config.threshold,
            min_distance_frames,
            ensure_positions=ensure_positions,
        )
        template_matches = matches_to_segments(peaks, hop, template_duration, label)
        _log(
            f"  detected {len(template_matches)} occurrence(s) "
            f"at threshold {match_config.threshold}:"
        )
        print_match_list(template_matches)
        matches.extend(template_matches)

        audit_peaks = pick_peaks(
            similarity,
            audit_floor,
            min_distance_frames,
            ensure_positions=ensure_positions,
        )
        audit_segments.extend(
            matches_to_segments(audit_peaks, hop, template_duration, label)
        )
        similarity_curves.append((label, similarity))
        per_template.append(
            {
                "template": label,
                "duration_seconds": round(template_duration, 6),
                "min_peak_distance_seconds": round(min_distance, 6),
                "matches": len(template_matches),
            }
        )

    matches.sort(key=lambda m: m.start)
    _log(
        f"total: {len(matches)} occurrence(s) across "
        f"{len(match_config.templates)} template(s)"
    )

    verification: dict | None = None
    if match_config.verify is not None:
        expected_start, expected_end = match_config.verify
        anchor = find_anchor_match(matches, expected_start, match_config.verify_tolerance_seconds)
        verification = {
            "expected_start": expected_start,
            "expected_end": expected_end,
            "tolerance_seconds": match_config.verify_tolerance_seconds,
            "ok": anchor is not None,
            "matched": anchor.as_dict() if anchor else None,
        }
        if anchor is not None:
            _log(
                f"verification OK: expected {format_timestamp(expected_start)}, "
                f"matched {format_timestamp(anchor.start)} (score {anchor.score:.4f})"
            )
        else:
            _log(
                f"verification FAILED: no match within "
                f"+/-{match_config.verify_tolerance_seconds:.1f}s of "
                f"{format_timestamp(expected_start)}"
            )

    dropped = [
        pad_segment(segment, match_config.pad_seconds, duration)
        for segment in select_dropped(matches, match_config.drop_all)
    ]
    dropped = merge_segments(dropped)
    removed, fraction, within_budget = check_removal_budget(
        dropped, duration, match_config.max_removed_fraction
    )
    keeps = complement_segments(dropped, duration)
    _log(
        f"removal: {len(dropped)} segment(s), {format_timestamp(removed)} "
        f"({fraction * 100:.2f}% of the file); keeps: {len(keeps)} segment(s)"
    )

    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "template_source": (
            str(template_source_path) if template_source_path is not None else None
        ),
        "duration_seconds": round(duration, 6),
        "dry_run": config.cut.dry_run,
        "feature_config": asdict(feature_config),
        "match_config": asdict(match_config),
        "templates": per_template,
        "verification": verification,
        "matches": [m.as_dict() for m in matches],
        "candidate_peaks": [m.as_dict() for m in audit_segments],
        "dropped": [s.as_dict() for s in dropped],
        "keeps": [s.as_dict() for s in keeps],
        "removed_seconds": round(removed, 6),
        "removed_fraction": round(fraction, 6),
        "within_budget": within_budget,
    }
    report_paths = write_reports(
        config.cut.report_dir,
        payload,
        similarity_curves=similarity_curves,
        frame_hop_seconds=hop,
    )
    _log(f"reports: {report_paths['matches_json'].parent}")

    if verification is not None and not verification["ok"] and not config.cut.force:
        _log("aborting before cut (use --force to override)")
        return 2
    if len(matches) < match_config.min_matches and not config.cut.force:
        _log(
            f"aborting: found {len(matches)} occurrence(s), fewer than "
            f"--min-matches {match_config.min_matches} (use --force to override)"
        )
        return 4
    if not within_budget and not config.cut.force:
        _log(
            f"aborting: removal exceeds --max-removed-fraction "
            f"{match_config.max_removed_fraction:.2f} (use --force to override)"
        )
        return 3

    if config.cut.dry_run:
        _log("dry run: no output written")
        return 0

    _log("cutting with ffmpeg -c copy (no re-encode)")

    def on_cut(index: int, count: int) -> None:
        _log(f"  segment {index}/{count}")

    cutter = LosslessCutter(
        input_path,
        work_dir=config.cut.work_dir,
        ffmpeg=config.cut.ffmpeg,
        keep_segments=config.cut.keep_segments,
    )
    cutter.cut(keeps, output_path, duration, progress=on_cut)
    new_duration = probe_duration(output_path, ffprobe=config.cut.ffprobe)
    _log(
        f"wrote {output_path} ({format_timestamp(new_duration)}, "
        f"removed {format_timestamp(duration - new_duration)})"
    )
    _log(f"done in {time.time() - started:.1f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except AudioCutError as error:
        print(f"{PREFIX} error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
