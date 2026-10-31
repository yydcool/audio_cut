"""Streaming frame feature extraction and feature caching.

Modules that follow ``FrameFeatureExtractor`` can be plugged into the pipeline
by registering them under a name; the CLI exposes those names via ``--feature``.

Frames are aligned to a *global* grid: frame ``f`` covers samples
``[f * hop_samples, f * hop_samples + n_fft)``. Alignment is kept across
decode-chunk boundaries, so matching results are independent of chunking.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft
from scipy.sparse import csr_matrix

from .audio_io import iter_decoded_chunks
from .config import AudioConfig, AudioCutError, FeatureConfig


# ---------------------------------------------------------------------------
# Feature extractor registry
# ---------------------------------------------------------------------------

_EXTRACTORS: dict[str, type["FrameFeatureExtractor"]] = {}


def register_extractor(name: str):
    def decorator(cls: type[FrameFeatureExtractor]) -> type[FrameFeatureExtractor]:
        _EXTRACTORS[name] = cls
        return cls

    return decorator


def build_extractor(config: FeatureConfig) -> "FrameFeatureExtractor":
    cls = _EXTRACTORS.get(config.name)
    if cls is None:
        known = ", ".join(sorted(_EXTRACTORS)) or "(none)"
        raise AudioCutError(f"unknown feature extractor '{config.name}'; known: {known}")
    return cls(config)


class FrameFeatureExtractor(ABC):
    """Turns fixed-size sample windows into feature vectors (one row per frame)."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_fft = int(config.n_fft)
        self.hop_samples = int(round(config.hop_seconds * config.sample_rate))
        if self.hop_samples <= 0:
            raise AudioCutError("hop_seconds too small for the chosen sample rate")
        self.hop_seconds = self.hop_samples / self.sample_rate
        self.window = np.hanning(self.n_fft).astype(np.float32)

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of values per frame."""

    @abstractmethod
    def extract_windows(self, windows: np.ndarray) -> np.ndarray:
        """``windows``: (n_frames, n_fft) -> features: (n_frames, n_features)."""


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sample_rate: int, fmin: float, fmax: float | None
) -> np.ndarray:
    """Triangular mel filters, shape (n_mels, n_freqs), rows normalised to unit sum."""
    n_freqs = n_fft // 2 + 1
    fmax = sample_rate / 2.0 if fmax is None else float(fmax)
    if not 0.0 <= fmin < fmax <= sample_rate / 2.0:
        raise AudioCutError(f"invalid mel frequency range {fmin}-{fmax} Hz")
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = edges[i : i + 3]
        rising = (freqs - left) / max(center - left, 1e-9)
        falling = (right - freqs) / max(right - center, 1e-9)
        bank[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-9)
    return bank


@register_extractor("logmel")
class LogMelExtractor(FrameFeatureExtractor):
    """Log-mel spectrogram: robust to level/encode changes, cheap, effective."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self.bank = mel_filterbank(
            config.n_mels, self.n_fft, self.sample_rate, config.fmin, config.fmax
        )
        # Only ~3% of the bank is nonzero (triangular filters); the sparse
        # product skips the zeros and is bitwise identical to the dense one,
        # which matters because numpy here links single-threaded reference BLAS.
        self.bank_sp = csr_matrix(self.bank)
        self.fft_workers = max(1, os.cpu_count() or 1)

    @property
    def n_features(self) -> int:
        return int(self.config.n_mels)

    def extract_windows(self, windows: np.ndarray) -> np.ndarray:
        frames = windows * self.window
        spectrum = rfft(frames, axis=1, workers=self.fft_workers)
        power = spectrum.real**2 + spectrum.imag**2
        mel = power @ self.bank_sp.T
        return np.log10(np.maximum(mel, self.config.log_floor)).astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming wrapper: global frame alignment across arbitrary chunks
# ---------------------------------------------------------------------------


class StreamFeaturizer:
    def __init__(self, extractor: FrameFeatureExtractor, max_batch_frames: int = 4096) -> None:
        self.extractor = extractor
        self.hop = extractor.hop_samples
        self.n_fft = extractor.n_fft
        self.max_batch = max_batch_frames
        self._buffer = np.empty(0, dtype=np.float32)
        self._base_sample = 0  # global sample index of _buffer[0]
        self._next_frame = 0  # next frame index to emit

    def feed(self, samples: np.ndarray) -> tuple[int, np.ndarray]:
        """Consume samples; return (start_frame_index, features) for new frames."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size:
            self._buffer = (
                np.concatenate((self._buffer, samples))
                if self._buffer.size
                else samples
            )
        first_frame = self._next_frame
        pieces: list[np.ndarray] = []
        while True:
            offset = self._next_frame * self.hop - self._base_sample
            available = self._buffer.size - offset - self.n_fft
            if available < 0:
                break
            batch = min(available // self.hop + 1, self.max_batch)
            if batch <= 0:
                break
            length = (batch - 1) * self.hop + self.n_fft
            windows = sliding_window_view(
                self._buffer[offset : offset + length], self.n_fft
            )[:: self.hop]
            pieces.append(self.extractor.extract_windows(windows))
            self._next_frame += batch
            new_base = self._next_frame * self.hop
            drop = new_base - self._base_sample
            if drop > 0:
                self._buffer = self._buffer[drop:]
                self._base_sample = new_base
        if not pieces:
            return first_frame, np.empty((0, self.extractor.n_features), dtype=np.float32)
        features = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        return first_frame, features


# ---------------------------------------------------------------------------
# Global statistics + normalisation
# ---------------------------------------------------------------------------


class FeatureStats:
    """Streaming per-feature mean/std over the whole file (float64 accumulators)."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.count = 0
        self._sum = np.zeros(n_features, dtype=np.float64)
        self._sumsq = np.zeros(n_features, dtype=np.float64)

    def update(self, features: np.ndarray) -> None:
        if features.size == 0:
            return
        as64 = features.astype(np.float64, copy=False)
        self.count += features.shape[0]
        self._sum += as64.sum(axis=0)
        self._sumsq += (as64 * as64).sum(axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise AudioCutError("no audio frames were extracted (empty input?)")
        mean = self._sum / self.count
        variance = np.maximum(self._sumsq / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"count": np.array([self.count]), "sum": self._sum, "sumsq": self._sumsq}

    @classmethod
    def from_arrays(cls, data) -> "FeatureStats":
        stats = cls(int(data["sum"].shape[0]))
        stats.count = int(data["count"][0])
        stats._sum = data["sum"].astype(np.float64)
        stats._sumsq = data["sumsq"].astype(np.float64)
        return stats


def normalize_features(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# End-to-end featurization + cache
# ---------------------------------------------------------------------------


def featurize_file(
    path: str | Path,
    extractor: FrameFeatureExtractor,
    audio_config: AudioConfig,
    ffmpeg: str = "ffmpeg",
    progress=None,
) -> tuple[np.ndarray, FeatureStats]:
    featurizer = StreamFeaturizer(extractor)
    stats = FeatureStats(extractor.n_features)
    pieces: list[np.ndarray] = []
    chunk_index = 0
    for chunk in iter_decoded_chunks(
        path,
        sample_rate=audio_config.sample_rate,
        chunk_seconds=audio_config.decode_chunk_seconds,
        ffmpeg=ffmpeg,
    ):
        chunk_index += 1
        _, features = featurizer.feed(chunk)
        if features.size:
            stats.update(features)
            pieces.append(features)
        if progress is not None:
            progress(chunk_index, chunk.size, pieces[-1].shape[0] if pieces else 0)
    if not pieces:
        raise AudioCutError(f"empty decode for {path}")
    return np.concatenate(pieces, axis=0), stats


def feature_cache_key(
    input_path: Path, feature_config: FeatureConfig, audio_config: AudioConfig
) -> str:
    stat = input_path.stat()
    payload = "|".join(
        str(part)
        for part in (
            input_path.name,
            stat.st_size,
            stat.st_mtime_ns,
            audio_config.sample_rate,
            feature_config.name,
            feature_config.n_fft,
            feature_config.hop_seconds,
            feature_config.n_mels,
            feature_config.fmin,
            feature_config.fmax,
            feature_config.log_floor,
        )
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def save_feature_cache(
    cache_dir: str | Path,
    key: str,
    features: np.ndarray,
    stats: FeatureStats,
) -> None:
    directory = Path(cache_dir) / key
    # Stage in a worker-unique directory, then atomically replace: concurrent
    # batch workers may compute the same key (shared --template-source) and
    # must never expose a half-written cache entry to each other.
    staging = (
        Path(cache_dir) / f"{key}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    np.save(staging / "features.npy", features)
    np.savez(staging / "stats.npz", **stats.to_arrays())
    try:
        os.replace(staging, directory)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)


def load_feature_cache(
    cache_dir: str | Path, key: str
) -> tuple[np.ndarray, FeatureStats] | None:
    directory = Path(cache_dir) / key
    features_file = directory / "features.npy"
    stats_file = directory / "stats.npz"
    if not features_file.is_file() or not stats_file.is_file():
        return None
    try:
        features = np.load(features_file)
        with np.load(stats_file) as data:
            stats = FeatureStats.from_arrays(data)
    except (OSError, ValueError, KeyError):
        return None
    return features, stats


def load_or_build_features(
    input_path: Path,
    extractor: FrameFeatureExtractor,
    audio_config: AudioConfig,
    cache_dir: str | Path | None,
    ffmpeg: str = "ffmpeg",
    progress=None,
) -> tuple[np.ndarray, FeatureStats, bool]:
    """Returns (features, stats, from_cache)."""
    key = None
    if cache_dir:
        key = feature_cache_key(input_path, extractor.config, audio_config)
        cached = load_feature_cache(cache_dir, key)
        if cached is not None:
            return cached[0], cached[1], True
    features, stats = featurize_file(
        input_path, extractor, audio_config, ffmpeg=ffmpeg, progress=progress
    )
    if cache_dir and key:
        save_feature_cache(cache_dir, key, features, stats)
    return features, stats, False
