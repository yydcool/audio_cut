"""Audio probing and streaming decode via ffmpeg/ffprobe subprocess pipes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import AudioCutError


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioCutError(f"required binary not found on PATH: {name}")
    return path


def probe_duration(path: str | Path, ffprobe: str = "ffprobe") -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioCutError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AudioCutError(f"could not parse duration of {path}: {exc}") from exc
    if duration <= 0:
        raise AudioCutError(f"non-positive duration reported for {path}")
    return duration


def _read_exact(stream, nbytes: int) -> bytes:
    parts: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        buf = stream.read(remaining)
        if not buf:
            break
        parts.append(buf)
        remaining -= len(buf)
    return b"".join(parts)


def iter_decoded_chunks(
    path: str | Path,
    sample_rate: int,
    chunk_seconds: float = 600.0,
    ffmpeg: str = "ffmpeg",
) -> Iterator[np.ndarray]:
    """Yield mono float32 sample chunks decoded from any ffmpeg-supported input.

    Streams through a pipe so a multi-hour source never has to fit in RAM as
    raw PCM; the consumer only keeps what it needs.
    """
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    bytes_per_chunk = int(round(chunk_seconds * sample_rate)) * 4
    try:
        while True:
            raw = _read_exact(proc.stdout, bytes_per_chunk)
            if not raw:
                break
            excess = len(raw) % 4
            if excess:
                raw = raw[:-excess]
            # Read-only view: downstream only reads (StreamFeaturizer copies
            # what it retains), so the extra 38 MB memcpy per chunk is skipped.
            yield np.frombuffer(raw, dtype="<f4")
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        returncode = proc.wait()
        if returncode != 0:
            raise AudioCutError(
                f"ffmpeg decode failed (exit {returncode}): {stderr[-2000:]}"
            )
