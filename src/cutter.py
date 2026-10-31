"""Lossless cutting: per-keep `ffmpeg -c copy` extraction + concat demuxer.

No audio is re-encoded, so quality is bit-preserved. Cut boundaries snap to
the container's packet/frame grid (approximately 23 ms for 44.1 kHz AAC),
which is inherent to stream copying.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import AudioCutError
from .segments import TimeSegment


class LosslessCutter:
    def __init__(
        self,
        input_path: str | Path,
        work_dir: str | Path,
        ffmpeg: str = "ffmpeg",
        keep_segments: bool = False,
    ) -> None:
        self.input_path = Path(input_path)
        self.work_dir = Path(work_dir)
        self.ffmpeg = ffmpeg
        self.keep_segments = keep_segments

    def _run(self, args: list[str], what: str) -> None:
        result = subprocess.run(
            [self.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", *args],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AudioCutError(
                f"{what} failed (exit {result.returncode}): {result.stderr.strip()[-2000:]}"
            )

    def _is_whole_file(self, keeps: list[TimeSegment], total_duration: float) -> bool:
        return (
            len(keeps) == 1
            and keeps[0].start <= 0.05
            and keeps[0].end >= total_duration - 0.5
        )

    def cut(
        self,
        keeps: list[TimeSegment],
        output_path: str | Path,
        total_duration: float,
        progress=None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not keeps:
            raise AudioCutError("nothing to keep; refusing to produce an empty file")

        if self._is_whole_file(keeps, total_duration):
            shutil.copy2(self.input_path, output_path)
            return output_path

        segments_dir = self.work_dir / "segments"
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
        segments_dir.mkdir(parents=True, exist_ok=True)

        segment_paths: list[Path] = []
        ffmpeg_args: list[str] = []
        for keep in keeps:
            segment_path = segments_dir / f"seg_{len(segment_paths):05d}{output_path.suffix}"
            ffmpeg_args += ["-ss", f"{keep.start:.6f}", "-i", str(self.input_path)]
            segment_paths.append(segment_path)
        for index, (keep, segment_path) in enumerate(zip(keeps, segment_paths)):
            ffmpeg_args += [
                "-map",
                f"{index}:a",
                "-t",
                f"{keep.duration:.6f}",
                "-vn",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(segment_path),
            ]
        self._run(ffmpeg_args, f"cutting {len(keeps)} keep segment(s)")
        for index in range(len(keeps)):
            if progress is not None:
                progress(index + 1, len(keeps))

        list_file = self.work_dir / "concat_list.txt"
        list_file.write_text(
            "".join(f"file '{_escape_concat_path(p.resolve())}'\n" for p in segment_paths),
            encoding="utf-8",
        )
        self._run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
                "-c",
                "copy",
                str(output_path),
            ],
            "concatenating keep segments",
        )

        if not self.keep_segments:
            shutil.rmtree(segments_dir, ignore_errors=True)
        return output_path


def _escape_concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")
