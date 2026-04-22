"""
Tier 1 tests for cxi_pipeline_ray.core.peak_finding.find_peaks_numpy.

These tests verify that peak finding (scipy.ndimage.label + center of mass)
recovers the ground-truth peak centers injected into synthetic logits by
tests.fixtures.make_synthetic_logits. No Ray, no writer, no GPU.

Class convention (matches cxi_pipeline_ray.core.peak_finding):
    class 0 = background (argmax here -> no peak)
    class 1 = peak (argmax here -> peak)
"""

from typing import List, Tuple

import numpy as np
import pytest

from cxi_pipeline_ray.core.peak_finding import find_peaks_numpy
from tests.fixtures import make_synthetic_logits


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _greedy_match(
    recovered: np.ndarray,
    ground_truth: List[Tuple[int, int]],
    tol: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Greedy nearest-neighbor match of recovered peaks to ground-truth coords.

    Args:
        recovered: (N, 3) array rows [_, y, x] from find_peaks_numpy.
        ground_truth: list of (y, x) integer tuples for the panel.
        tol: max pixel distance for a match to be accepted.

    Returns:
        List of (recovered_index, ground_truth_index) pairs actually matched.

    Asserts:
        - Every recovered peak matches some unused ground-truth peak within tol.
        - Every ground-truth peak is matched by some recovered peak.
    """
    assert recovered.shape[0] == len(ground_truth), (
        f"peak count mismatch: recovered={recovered.shape[0]} "
        f"ground_truth={len(ground_truth)}"
    )

    used_gt = set()
    pairs: List[Tuple[int, int]] = []
    for rec_idx in range(recovered.shape[0]):
        ry, rx = recovered[rec_idx, 1], recovered[rec_idx, 2]
        best_gt = -1
        best_dist = float("inf")
        for gt_idx, (gy, gx) in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue
            dist = ((ry - gy) ** 2 + (rx - gx) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_gt = gt_idx
        assert best_gt >= 0, f"no ground-truth candidate for recovered peak {rec_idx}"
        assert best_dist <= tol, (
            f"recovered peak {rec_idx} at ({ry:.3f},{rx:.3f}) is "
            f"{best_dist:.3f}px from nearest ground-truth "
            f"(tolerance={tol})"
        )
        used_gt.add(best_gt)
        pairs.append((rec_idx, best_gt))
    return pairs


# -----------------------------------------------------------------------------
# (a) zero peaks on every panel -> (0, 3) output
# -----------------------------------------------------------------------------

def test_zero_peaks():
    """find_peaks_numpy returns an (0, 3) array when the panel has no peaks."""
    B, C, H, W = 1, 4, 64, 64
    logits, gt = make_synthetic_logits(
        B=B, C=C, H=H, W=W, peak_locations=[[] for _ in range(B * C)], seed=0
    )
    assert logits.shape == (B * C, 2, H, W)
    for panel_idx in range(B * C):
        peaks = find_peaks_numpy(logits[panel_idx])
        assert peaks.shape == (0, 3), (
            f"panel {panel_idx}: expected shape (0, 3), got {peaks.shape}"
        )
        assert len(gt[panel_idx]) == 0


# -----------------------------------------------------------------------------
# (b) single peak at panel center
# -----------------------------------------------------------------------------

def test_single_peak_center():
    """A single peak at (H//2, W//2) on panel 0 is recovered within 1px."""
    B, C, H, W = 1, 4, 64, 64
    center = (H // 2, W // 2)
    peak_locations = [[center]] + [[] for _ in range(B * C - 1)]
    logits, gt = make_synthetic_logits(
        B=B, C=C, H=H, W=W, peak_locations=peak_locations, peak_blob_size=3, seed=0
    )

    peaks0 = find_peaks_numpy(logits[0])
    _greedy_match(peaks0, gt[0], tol=1.0)

    # Other panels are empty.
    for panel_idx in range(1, B * C):
        peaks = find_peaks_numpy(logits[panel_idx])
        assert peaks.shape == (0, 3)


# -----------------------------------------------------------------------------
# (c) many peaks per panel -> all recovered within 1px
# -----------------------------------------------------------------------------

def test_many_peaks():
    """peaks_per_panel=10 with H=W=128 and peak_blob_size=3 — all 10 recovered."""
    B, C, H, W = 1, 4, 128, 128
    logits, gt = make_synthetic_logits(
        B=B,
        C=C,
        H=H,
        W=W,
        peaks_per_panel=10,
        peak_blob_size=3,
        seed=42,
    )
    for panel_idx in range(B * C):
        assert len(gt[panel_idx]) == 10, (
            f"panel {panel_idx}: ground truth has "
            f"{len(gt[panel_idx])} peaks, expected 10"
        )
        peaks = find_peaks_numpy(logits[panel_idx])
        _greedy_match(peaks, gt[panel_idx], tol=1.0)


# -----------------------------------------------------------------------------
# (d) parametrize over several panel shapes
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "B,C,H,W",
    [
        (1, 4, 128, 128),
        (2, 8, 256, 256),
        (1, 1, 64, 64),
    ],
)
def test_varying_dims(B, C, H, W):
    """Single-peak-at-center recovery works across several input shapes."""
    center = (H // 2, W // 2)
    peak_locations = [[center] for _ in range(B * C)]
    logits, gt = make_synthetic_logits(
        B=B, C=C, H=H, W=W, peak_locations=peak_locations, peak_blob_size=3, seed=0
    )
    assert logits.shape == (B * C, 2, H, W)
    for panel_idx in range(B * C):
        peaks = find_peaks_numpy(logits[panel_idx])
        _greedy_match(peaks, gt[panel_idx], tol=1.0)


# -----------------------------------------------------------------------------
# (e) return_seg_map contract
# -----------------------------------------------------------------------------

def test_return_seg_map():
    """With return_seg_map=True, seg_map is (H, W) uint8 with values {0, 1}."""
    B, C, H, W = 1, 1, 64, 64
    logits, _ = make_synthetic_logits(
        B=B, C=C, H=H, W=W, peaks_per_panel=3, peak_blob_size=3, seed=0
    )
    peaks, seg_map = find_peaks_numpy(logits[0], return_seg_map=True)

    assert seg_map.dtype == np.uint8, f"expected uint8, got {seg_map.dtype}"
    assert seg_map.shape == (H, W), f"expected ({H}, {W}), got {seg_map.shape}"

    unique_values = set(np.unique(seg_map).tolist())
    assert unique_values <= {0, 1}, (
        f"seg_map contains unexpected values: {unique_values}"
    )
    # Sanity: we asked for 3 peaks so seg_map should contain BOTH classes.
    assert unique_values == {0, 1}, (
        f"expected seg_map to contain both 0 and 1, got {unique_values}"
    )

    # Peaks still returned alongside the map.
    assert peaks.shape[1] == 3
