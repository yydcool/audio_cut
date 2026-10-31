#!/usr/bin/env python3
"""Batch runner: apply the audio_cut pipeline to every audio file in a directory.

Files are processed sequentially; a failure or a guard trip on one file does
not stop the batch. Originals are left untouched: outputs land in
<output-dir>/ (default: <input-dir>/cleaned/) together with per-file audit
reports and a batch_summary.json.

By default each file's own --template regions are used (pipeline default 0-26).
Pass --template-source to take the patterns from one reference file instead,
so files whose pattern sits at a different offset are still matched; without
an explicit --template the regions default to 0-26 and 06:50-06:59.

Usage:
    python3 batch_process.py --input-dir download
    python3 batch_process.py --input-dir download --template-source audio.m4a
    python3 batch_process.py --input-dir download --template-source audio.m4a --jobs 4
    python3 batch_process.py --input-dir download --min-occurrences 2
    python3 batch_process.py --input-dir download --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from src.__main__ import main as audio_cut_main
from src.report import format_timestamp

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4b",
    ".mka",
    ".mov",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}

EXIT_LABELS = {
    0: "ok",
    1: "failed",
    2: "failed: verify anchor not found",
    3: "failed: removal budget exceeded",
    4: "skipped: too few occurrences",
}

DEFAULT_SOURCE_TEMPLATES = ["0-26", "06:50-06:59"]


def template_regions(args: argparse.Namespace) -> list[str]:
    if args.template:
        return list(args.template)
    if args.template_source is not None:
        return list(DEFAULT_SOURCE_TEMPLATES)
    return []


@dataclass(slots=True)
class FileResult:
    source: str
    status: str
    exit_code: int | None = None
    output: str | None = None
    occurrences: int | None = None
    removed_seconds: float | None = None
    duration_seconds: float | None = None
    seconds: float = 0.0
    detail: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch_process.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="download",
        help="directory containing source audio files (default: download)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: <input-dir>/cleaned)",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=None,
        help=(
            "template region, repeatable for multiple patterns, e.g. "
            "--template 0-26 --template 06:50-06:59 "
            "(default: pipeline default 0-26)"
        ),
    )
    parser.add_argument(
        "--template-source",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "read the template regions from this reference file instead of "
            "each input file; without an explicit --template the regions "
            "default to 0-26 and 06:50-06:59"
        ),
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="skip a file when fewer matches than this are found (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="NCC threshold (default: pipeline default)",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=None,
        help="extra margin around removals, seconds (default: pipeline default)",
    )
    parser.add_argument(
        "--max-removed-fraction",
        type=float,
        default=None,
        help="removal budget (default: pipeline default)",
    )
    parser.add_argument(
        "--verify",
        default=None,
        metavar="START-END",
        help="optional per-file anchor; only meaningful if all files share a timeline",
    )
    parser.add_argument(
        "--keep-first",
        action="store_true",
        help="keep the first occurrence instead of dropping all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="detect and report only; do not cut",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reprocess files whose output already exists",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "process up to N files in parallel threads (default: 1). Decode "
            "runs in ffmpeg subprocesses and numpy releases the GIL, so this "
            "scales with core count; progress lines may interleave"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the feature cache",
    )
    return parser


def discover_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def process_one(
    source: Path,
    output: Path,
    report_dir: Path,
    work_dir: Path,
    args: argparse.Namespace,
) -> FileResult:
    started = time.time()
    argv = [
        "--input",
        str(source),
        "--output",
        str(output),
        "--report-dir",
        str(report_dir),
        "--work-dir",
        str(work_dir),
        "--min-matches",
        str(args.min_occurrences),
    ]
    for template in template_regions(args):
        argv += ["--template", template]
    if args.template_source is not None:
        argv += ["--template-source", str(args.template_source)]
    for flag, value in (
        ("--threshold", args.threshold),
        ("--pad", args.pad),
        ("--max-removed-fraction", args.max_removed_fraction),
        ("--verify", args.verify),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if args.keep_first:
        argv.append("--keep-first")
    if args.dry_run:
        argv.append("--dry-run")
    if args.no_cache:
        argv.append("--no-cache")

    code = audio_cut_main(argv)

    occurrences = removed = duration = None
    matches_json = report_dir / "matches.json"
    if matches_json.is_file():
        try:
            payload = json.loads(matches_json.read_text(encoding="utf-8"))
            occurrences = len(payload.get("matches", []))
            removed = payload.get("removed_seconds")
            duration = payload.get("duration_seconds")
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            pass
    if code != 0:
        removed = None

    status = EXIT_LABELS.get(code, f"failed: exit {code}")
    detail = ""
    if code == 0 and args.dry_run:
        status = "ok (dry-run)"
    if code == 4:
        detail = f"only {occurrences} occurrence(s) found"
    if code == 0 and not args.dry_run:
        shutil.rmtree(work_dir, ignore_errors=True)

    return FileResult(
        source=str(source),
        status=status,
        exit_code=code,
        output=str(output) if code == 0 and not args.dry_run else None,
        occurrences=occurrences,
        removed_seconds=removed,
        duration_seconds=duration,
        seconds=time.time() - started,
        detail=detail,
    )


def print_summary(results: list[FileResult]) -> None:
    if not results:
        print("\nno files completed")
        return
    name_width = min(max(len(Path(r.source).name) for r in results), 60)
    statuses = [
        result.status + (f" ({result.detail})" if result.detail else "")
        for result in results
    ]
    status_width = min(max(len(status) for status in statuses), 70)
    print()
    print(
        f"{'file':<{name_width}}  {'status':<{status_width}} "
        f"{'occ':>3} {'removed':>10}  {'time':>8}"
    )
    for result, status in zip(results, statuses):
        removed = (
            format_timestamp(result.removed_seconds) if result.removed_seconds else "-"
        )
        occurrences = "-" if result.occurrences is None else str(result.occurrences)
        print(
            f"{Path(result.source).name:<{name_width}}  {status:<{status_width}} "
            f"{occurrences:>3} {removed:>10}  {result.seconds:7.1f}s"
        )
    total_removed = sum(result.removed_seconds or 0.0 for result in results)
    print(f"\ntotal removed: {format_timestamp(total_removed)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: input directory not found: {input_dir}", file=sys.stderr)
        return 1
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "cleaned"
    )
    if output_dir == input_dir:
        print("error: --output-dir must differ from --input-dir", file=sys.stderr)
        return 1
    if args.template_source is not None:
        args.template_source = Path(args.template_source).expanduser().resolve()
        if not args.template_source.is_file():
            print(
                f"error: template source not found: {args.template_source}",
                file=sys.stderr,
            )
            return 1
    files = discover_audio_files(input_dir)
    if not files:
        print(f"error: no audio files found in {input_dir}", file=sys.stderr)
        return 1

    if args.jobs < 1:
        print("error: --jobs must be >= 1", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_root = output_dir / "reports"
    print(f"batch: {len(files)} file(s), {input_dir} -> {output_dir} (jobs={args.jobs})")

    tasks: list[tuple[int, Path, Path, Path, Path]] = []
    by_index: dict[int, FileResult] = {}
    for index, source in enumerate(files, start=1):
        output = output_dir / source.name
        if output.exists() and not args.overwrite:
            by_index[index] = FileResult(
                source=str(source), status="skipped: output exists"
            )
            print(
                f"[{index}/{len(files)}] {source.name}: "
                "skipped (output exists; use --overwrite to redo)"
            )
        else:
            tasks.append(
                (
                    index,
                    source,
                    output,
                    reports_root / f"{source.name}.report",
                    reports_root / f"{source.name}.work",
                )
            )

    failures = 0
    interrupted = False
    started = time.time()

    def finish(index: int, source: Path, result: FileResult) -> None:
        nonlocal failures
        by_index[index] = result
        print(
            f"[{index}/{len(files)}] {source.name}: {result.status}"
            + (f" - {result.detail}" if result.detail else "")
        )
        if result.status.startswith("failed"):
            failures += 1

    def run_task(task: tuple[int, Path, Path, Path, Path]) -> FileResult:
        index, source, output, report_dir, work_dir = task
        try:
            return process_one(source, output, report_dir, work_dir, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return FileResult(
                source=str(source),
                status=f"failed: crash ({type(exc).__name__})",
                detail=str(exc)[:200],
            )

    if args.jobs == 1:
        try:
            for task in tasks:
                print(f"[{task[0]}/{len(files)}] {task[1].name}: processing")
                finish(task[0], task[1], run_task(task))
        except KeyboardInterrupt:
            interrupted = True
            print("\nbatch interrupted by user", file=sys.stderr)
    else:
        pool = ThreadPoolExecutor(max_workers=args.jobs)
        pending = {pool.submit(run_task, task): task for task in tasks}
        try:
            for future in as_completed(pending):
                index, source, _, _, _ = pending[future]
                finish(index, source, future.result())
        except KeyboardInterrupt:
            interrupted = True
            print("\nbatch interrupted by user", file=sys.stderr)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    results = [by_index[index] for index in sorted(by_index)]

    print_summary(results)
    reports_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "template": args.template,
        "template_source": (
            str(args.template_source) if args.template_source is not None else None
        ),
        "jobs": args.jobs,
        "min_occurrences": args.min_occurrences,
        "dry_run": args.dry_run,
        "elapsed_seconds": round(time.time() - started, 3),
        "files": [asdict(result) for result in results],
    }
    summary_path = reports_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")

    if interrupted:
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
