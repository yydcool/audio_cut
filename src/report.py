"""Audit artifacts: matches.json + scores.csv + console summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .segments import TimeSegment


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{remainder:06.3f}"
    return f"{minutes:02d}:{remainder:06.3f}"


def write_reports(
    report_dir: str | Path,
    payload: dict,
    similarity_curves: list[tuple[str, np.ndarray]] | None = None,
    frame_hop_seconds: float | None = None,
) -> dict[str, Path]:
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "matches.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    paths = {"matches_json": json_path}
    if similarity_curves and frame_hop_seconds:
        multiple = len(similarity_curves) > 1
        for index, (_, similarity) in enumerate(similarity_curves, start=1):
            name = f"scores_{index}.csv" if multiple else "scores.csv"
            csv_path = directory / name
            times = np.arange(similarity.shape[0], dtype=np.float64) * frame_hop_seconds
            table = np.column_stack([times, similarity.astype(np.float64)])
            np.savetxt(
                csv_path,
                table,
                delimiter=",",
                header="time_seconds,similarity",
                comments="",
                fmt="%.3f,%.6f",
            )
            paths[f"scores_{index}"] = csv_path
    return paths


def print_match_list(matches: list[TimeSegment]) -> None:
    if not matches:
        print("[audio_cut]   (no occurrences detected)")
        return
    width = len(str(len(matches)))
    for index, match in enumerate(matches, start=1):
        score = "-" if match.score is None else f"{match.score:.4f}"
        print(
            f"[audio_cut]   {index:>{width}}. "
            f"{format_timestamp(match.start)} - {format_timestamp(match.end)}  "
            f"score {score}"
        )
