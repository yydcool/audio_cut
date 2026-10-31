"""Central configuration dataclasses and shared error type."""

from __future__ import annotations

from dataclasses import dataclass, field


class AudioCutError(Exception):
    """Fatal, user-facing error. Message is printed without traceback."""


@dataclass(slots=True)
class AudioConfig:
    """Decoding parameters (ffmpeg pipe)."""

    sample_rate: int = 16000
    decode_chunk_seconds: float = 600.0


@dataclass(slots=True)
class FeatureConfig:
    """Analysis window / spectrogram parameters."""

    name: str = "logmel"
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_seconds: float = 0.1
    n_mels: int = 64
    fmin: float = 30.0
    fmax: float | None = None  # None -> sample_rate / 2
    log_floor: float = 1e-10


@dataclass(slots=True)
class MatchConfig:
    """Template matching parameters.

    templates          : list of (start, end) regions whose occurrences are all
                         removed in one pass, e.g. [(0, 26), (410, 419)]
    threshold          : minimum NCC similarity for a peak to count as a match
    min_peak_distance_seconds : minimum spacing between accepted peaks
                                (None -> 0.9 * template duration)
    drop_all           : True  -> remove every occurrence, including the original
                         False -> keep the first occurrence, drop the rest
    pad_seconds        : expanded margin around each detected occurrence
    max_removed_fraction : safety guard; abort if more than this share would be cut
    min_matches        : safety guard; abort if fewer occurrences were detected
    verify             : optional (start, end) anchor, e.g. (531, 557) for 08:51-09:17
    verify_tolerance_seconds : accepted start offset for the anchor
    """

    templates: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 26.0)]
    )
    threshold: float = 0.85
    min_peak_distance_seconds: float | None = None
    drop_all: bool = True
    pad_seconds: float = 0.2
    max_removed_fraction: float = 0.15
    min_matches: int = 1
    verify: tuple[float, float] | None = None
    verify_tolerance_seconds: float = 2.0


@dataclass(slots=True)
class CutConfig:
    """Lossless cutting parameters."""

    output_path: str
    work_dir: str
    report_dir: str
    dry_run: bool = False
    force: bool = False
    keep_segments: bool = False
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(slots=True)
class RunConfig:
    """Aggregate used to pass everything around as a single object."""

    input_path: str
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    cut: CutConfig = field(default_factory=lambda: CutConfig("", "", ""))
