"""Template matchers: locate occurrences of a template within a feature matrix.

The default matcher uses normalised cross-correlation (NCC) between the
template's log-mel patch and every sliding window of the file. NCC is the
standard approach for spectrogram template matching (librosa/scikit-maad/ohun
all build on the same idea): shift- and level-invariant, robust to re-encoding.

New algorithms (e.g. landmark fingerprints) can be added by subclassing
``Matcher`` and registering under a name; the CLI selects them via ``--matcher``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import fftconvolve, find_peaks

from .config import AudioCutError

_MATCHERS: dict[str, type["Matcher"]] = {}


def register_matcher(name: str):
    def decorator(cls: type[Matcher]) -> type[Matcher]:
        _MATCHERS[name] = cls
        return cls

    return decorator


def build_matcher(name: str) -> "Matcher":
    cls = _MATCHERS.get(name)
    if cls is None:
        known = ", ".join(sorted(_MATCHERS)) or "(none)"
        raise AudioCutError(f"unknown matcher '{name}'; known: {known}")
    return cls()


class Matcher(ABC):
    @abstractmethod
    def similarity(self, template: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Return a similarity curve: ``sim[i]`` scores a window starting at frame i."""


def sliding_window_sum(values: np.ndarray, width: int) -> np.ndarray:
    """Sum of ``values`` over every consecutive window of ``width`` samples."""
    cumulative = np.cumsum(values, dtype=np.float64)
    return cumulative[width - 1 :] - np.concatenate(([0.0], cumulative[:-width]))


@register_matcher("ncc")
class NCCTemplateMatcher(Matcher):
    """Pearson correlation between template patch (T x B) and every window.

    score(i) = <template - mean(template), window_i> / (||template|| * ||window_i||)
    with the window mean/norm computed from sliding sums in O(n) time; the
    sliding dot products are evaluated with FFT convolution per mel band.
    """

    def similarity(self, template: np.ndarray, features: np.ndarray) -> np.ndarray:
        template = np.asarray(template, dtype=np.float64)
        features = np.asarray(features, dtype=np.float64)
        t_frames, n_bands = template.shape
        n_frames = features.shape[0]
        if t_frames == 0 or n_bands == 0:
            raise AudioCutError("template is empty")
        if n_frames < t_frames:
            return np.zeros(0, dtype=np.float32)

        centered = template - template.mean()
        template_energy = float((centered * centered).sum())
        if template_energy <= 1e-9:
            raise AudioCutError("template has no variance (silent/constant region?)")

        n_elements = float(t_frames * n_bands)
        row_sum = features.sum(axis=1)
        row_sq = (features * features).sum(axis=1)
        window_sum = sliding_window_sum(row_sum, t_frames)
        window_sq = sliding_window_sum(row_sq, t_frames)
        window_energy = np.maximum(window_sq - window_sum * window_sum / n_elements, 0.0)

        flipped = centered[::-1, :]
        dot = np.zeros(n_frames - t_frames + 1, dtype=np.float64)
        for band in range(n_bands):
            dot += fftconvolve(features[:, band], flipped[:, band], mode="valid")

        denominator = np.sqrt(template_energy * window_energy)
        similarity = np.zeros_like(dot)
        np.divide(dot, denominator, out=similarity, where=denominator > 1e-9)
        return similarity.astype(np.float32)


def pick_peaks(
    similarity: np.ndarray,
    threshold: float,
    min_distance_frames: int,
    ensure_positions: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Local maxima of the similarity curve above ``threshold``.

    ``min_distance_frames`` suppresses multiple peaks inside one occurrence.
    ``ensure_positions`` forces specific window starts to be considered even
    when they sit at the array boundary (``find_peaks`` cannot see index 0);
    this is how the template's own location is kept in the drop list.
    Returns (frame_index, score) sorted by time.
    """
    if similarity.size == 0:
        return []
    distance = max(1, int(min_distance_frames))
    indices, properties = find_peaks(
        similarity.astype(np.float32),
        height=threshold,
        distance=distance,
    )
    accepted = {int(i): float(h) for i, h in zip(indices, properties["peak_heights"])}
    for position in ensure_positions or []:
        if position < 0 or position >= similarity.size:
            continue
        score = float(similarity[position])
        if score < threshold:
            continue
        if any(abs(position - frame) < distance for frame in accepted):
            continue
        accepted[position] = score
    return sorted(accepted.items())
