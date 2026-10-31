"""Time-domain segment algebra: padding, merging, keep/drop computation, guards."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TimeSegment:
    start: float
    end: float
    score: float | None = None
    template: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration": round(self.duration, 6),
            "score": None if self.score is None else round(self.score, 6),
            "template": self.template,
        }


def matches_to_segments(
    peaks: list[tuple[int, float]],
    frame_hop_seconds: float,
    template_duration: float,
    template: str | None = None,
) -> list[TimeSegment]:
    """Convert (start_frame, score) peaks into time segments covering the match."""
    segments = [
        TimeSegment(
            start=frame * frame_hop_seconds,
            end=frame * frame_hop_seconds + template_duration,
            score=score,
            template=template,
        )
        for frame, score in peaks
    ]
    segments.sort(key=lambda s: s.start)
    return segments


def pad_segment(segment: TimeSegment, pad_seconds: float, duration: float) -> TimeSegment:
    return TimeSegment(
        start=max(0.0, segment.start - pad_seconds),
        end=min(duration, segment.end + pad_seconds),
        score=segment.score,
        template=segment.template,
    )


def merge_segments(segments: list[TimeSegment], max_gap: float = 0.0) -> list[TimeSegment]:
    """Merge overlapping or nearly-touching segments; keep the best score."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start)
    merged = [
        TimeSegment(ordered[0].start, ordered[0].end, ordered[0].score, ordered[0].template)
    ]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start <= last.end + max_gap:
            last.end = max(last.end, segment.end)
            if segment.score is not None and (last.score is None or segment.score > last.score):
                last.score = segment.score
                last.template = segment.template
        else:
            merged.append(
                TimeSegment(segment.start, segment.end, segment.score, segment.template)
            )
    return merged


def select_dropped(matches: list[TimeSegment], drop_all: bool) -> list[TimeSegment]:
    """Choose which occurrences to remove.

    drop_all=True  -> every occurrence, including the original.
    drop_all=False -> keep the first occurrence, drop the rest.
    """
    if not matches:
        return []
    ordered = sorted(matches, key=lambda s: s.start)
    return list(ordered) if drop_all else list(ordered[1:])


def complement_segments(
    segments: list[TimeSegment], duration: float, min_keep_duration: float = 0.05
) -> list[TimeSegment]:
    """Invert drop segments into the keep segments to retain."""
    keeps: list[TimeSegment] = []
    cursor = 0.0
    for segment in sorted(segments, key=lambda s: s.start):
        if segment.start > cursor:
            keeps.append(TimeSegment(cursor, min(segment.start, duration)))
        cursor = max(cursor, segment.end)
    if cursor < duration:
        keeps.append(TimeSegment(cursor, duration))
    return [keep for keep in keeps if keep.duration >= min_keep_duration]


def total_duration(segments: list[TimeSegment]) -> float:
    return float(sum(segment.duration for segment in segments))


def check_removal_budget(
    dropped: list[TimeSegment], duration: float, max_fraction: float
) -> tuple[float, float, bool]:
    """Returns (removed_seconds, removed_fraction, within_budget)."""
    if duration <= 0:
        return 0.0, 0.0, True
    removed = total_duration(dropped)
    fraction = removed / duration
    return removed, fraction, fraction <= max_fraction


def find_anchor_match(
    matches: list[TimeSegment], expected_start: float, tolerance: float
) -> TimeSegment | None:
    """Closest match whose start lies within ``tolerance`` of ``expected_start``."""
    candidates = [m for m in matches if abs(m.start - expected_start) <= tolerance]
    if not candidates:
        return None
    return min(candidates, key=lambda m: abs(m.start - expected_start))
