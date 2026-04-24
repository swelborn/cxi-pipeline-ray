"""
Tier 1 end-to-end writer tests for cxi_pipeline_ray.

Drives a single CXIFileWriterActor through the full
process_batch -> flush -> CXI write path using only the synthetic logits
fixture from tests.fixtures. No Q2 queue, no GPU inference, no LCLS data.

Class convention (matches cxi_pipeline_ray.core.peak_finding):
    class 0 = background
    class 1 = peak

Note: the writer requires a real detector image per event to assemble and
write to /entry_1/data_1/data. FakePipelineOutput(logits) alone gives
images=[None] and the actor's submit_processed_batch raises an
AttributeError on ``img.shape`` (see file_writer.py:194), so events never
land in the buffer. To exercise the writer output path we attach a minimal
preprocessing_metadata (identity shape - no padding) and a ray.put(...) of
tiny detector images. Per the task spec we do NOT assert on
/entry_1/data_1/data here; that belongs to Tier 2 (Task 18).
"""

from types import SimpleNamespace
from typing import List, Tuple

import glob
import os
import tempfile

import h5py
import numpy as np
import pytest
import ray

from cxi_pipeline_ray.core.coordinator import process_batch
from cxi_pipeline_ray.core.file_writer import CXIFileWriterActor
from tests.fixtures import FakePipelineOutput, make_synthetic_logits


# -----------------------------------------------------------------------------
# Ray lifecycle - module-scoped so all tests share one local cluster.
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ray_cluster():
    """
    Start a small local Ray cluster for the duration of this module.

    ``_temp_dir`` is deliberately short because Ray creates a plasma store
    AF_UNIX socket inside it, and Linux caps socket paths at 108 bytes.
    A long prefix like ``/lscratch/cwang31/tmp/ray-test-writer-offline-XXXX``
    already blows the budget before the session/sockets segments are added.
    """
    tmp = tempfile.mkdtemp(prefix="rt-", dir="/tmp")
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=tmp,
    )
    yield
    ray.shutdown()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_pipeline_output(
    B: int,
    C: int,
    H: int,
    W: int,
    peaks_per_panel: int = 3,
    peak_locations=None,
    seed: int = 0,
    photon_wavelength: float = 1.3,
    timestamp: int = 0,
):
    """
    Build a FakePipelineOutput with just enough structure for the writer to
    run end-to-end. Uses identity preprocessing (original_shape ==
    preprocessed_shape) so the coordinator performs no unpadding and no
    peak clipping beyond y<H, x<W.

    Returns:
        (pipeline_output, ground_truth) where ground_truth is a list of
        length B*C of [(y, x), ...] per panel.
    """
    logits, ground_truth = make_synthetic_logits(
        B=B,
        C=C,
        H=H,
        W=W,
        peaks_per_panel=peaks_per_panel,
        peak_locations=peak_locations,
        seed=seed,
    )

    # Detector images: (B*C, 1, H, W) to match reconstruct_from_arrays
    # contract. Content is irrelevant here - Tier 1 does not assert on
    # /entry_1/data_1/data.
    detector_images = np.zeros((B * C, 1, H, W), dtype=np.float32)
    original_image_ref = ray.put(detector_images)

    preprocessing_metadata = SimpleNamespace(
        original_shape=(B, C, H, W),
        preprocessed_shape=(B * C, 1, H, W),
    )

    metadata = {"photon_wavelength": photon_wavelength, "timestamp": timestamp}

    pipeline_output = FakePipelineOutput(
        logits=logits,
        preprocessing_metadata=preprocessing_metadata,
        original_image_ref=original_image_ref,
        metadata=metadata,
    )
    return pipeline_output, ground_truth


def _make_writer(
    tmp_path,
    buffer_size: int,
    min_num_peak: int = 1,
    max_num_peak: int = 1024,
):
    return CXIFileWriterActor.remote(
        output_dir=str(tmp_path),
        geom_file=None,
        buffer_size=buffer_size,
        min_num_peak=min_num_peak,
        max_num_peak=max_num_peak,
        file_prefix="test",
        crystfel_mode=False,
        save_segmentation_maps=False,
    )


def _list_cxi_files(tmp_path) -> List[str]:
    return sorted(glob.glob(os.path.join(str(tmp_path), "*.cxi")))


def _greedy_match_peaks(
    recovered_y: np.ndarray,
    recovered_x: np.ndarray,
    ground_truth: List[Tuple[int, int]],
    tol: float = 1.0,
) -> None:
    """
    Greedy nearest-neighbor match of recovered (y, x) coords to ground-truth
    peaks. Asserts every recovered peak matches some unused ground-truth
    peak within ``tol`` and every ground-truth peak is matched.
    """
    assert len(recovered_y) == len(ground_truth), (
        f"peak count mismatch: recovered={len(recovered_y)} "
        f"ground_truth={len(ground_truth)}"
    )

    used = set()
    for rec_idx in range(len(recovered_y)):
        ry = float(recovered_y[rec_idx])
        rx = float(recovered_x[rec_idx])
        best_gt = -1
        best_dist = float("inf")
        for gt_idx, (gy, gx) in enumerate(ground_truth):
            if gt_idx in used:
                continue
            dist = ((ry - gy) ** 2 + (rx - gx) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_gt = gt_idx
        assert best_gt >= 0, f"no candidate for recovered peak {rec_idx}"
        assert best_dist <= tol, (
            f"recovered peak {rec_idx} at ({ry:.3f}, {rx:.3f}) is "
            f"{best_dist:.3f}px from nearest ground-truth "
            f"(tolerance={tol})"
        )
        used.add(best_gt)


# -----------------------------------------------------------------------------
# (a) single batch with one-event buffer -> exactly one CXI file
# -----------------------------------------------------------------------------

def test_single_batch_one_file(ray_cluster, tmp_path):
    """One batch (B=1, C=4) with buffer_size=1 flushes a single CXI file."""
    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    pipeline_output, _gt = _make_pipeline_output(
        B=1, C=4, H=64, W=64, peaks_per_panel=3, seed=0,
    )
    process_batch(pipeline_output, writer)

    stats = ray.get(writer.flush_final.remote())
    assert stats["total_events_written"] == 1
    assert stats["total_events_filtered"] == 0

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1, f"expected 1 .cxi file, got {len(cxi_files)}"

    with h5py.File(cxi_files[0], "r") as f:
        n_peaks = f["/entry_1/result_1/nPeaks"][()]
        assert n_peaks.shape == (1,), f"nPeaks shape {n_peaks.shape}"


# -----------------------------------------------------------------------------
# (b) buffer threshold: 3 batches with buffer_size=3 -> one file, 3 events
# -----------------------------------------------------------------------------

def test_buffer_flush_threshold(ray_cluster, tmp_path):
    """With buffer_size=3, three B=1 batches flush as a single 3-event file."""
    writer = _make_writer(tmp_path, buffer_size=3, min_num_peak=1)

    for batch_idx in range(3):
        pipeline_output, _gt = _make_pipeline_output(
            B=1, C=4, H=64, W=64, peaks_per_panel=3, seed=batch_idx,
        )
        process_batch(pipeline_output, writer)

    stats = ray.get(writer.flush_final.remote())
    assert stats["total_events_written"] == 3
    assert stats["chunks_written"] == 1
    assert stats["total_events_filtered"] == 0

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1, f"expected 1 .cxi file, got {len(cxi_files)}"

    with h5py.File(cxi_files[0], "r") as f:
        n_peaks = f["/entry_1/result_1/nPeaks"][()]
        assert n_peaks.shape == (3,), f"nPeaks shape {n_peaks.shape}"


# -----------------------------------------------------------------------------
# (c) min_num_peak filters events with zero peaks -> no CXI file written
# -----------------------------------------------------------------------------

def test_min_num_peak_filters_event(ray_cluster, tmp_path):
    """Zero-peak events with min_num_peak=1 are filtered; nothing is written."""
    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    # Empty peak lists for every panel in a B=1, C=4 batch.
    peak_locations = [[] for _ in range(1 * 4)]
    pipeline_output, _gt = _make_pipeline_output(
        B=1, C=4, H=64, W=64, peak_locations=peak_locations,
    )
    process_batch(pipeline_output, writer)

    stats = ray.get(writer.flush_final.remote())
    # Either zero files or a file with zero events is acceptable per spec.
    # Current behavior (verified): filtered event stays out of buffer, and
    # _flush_buffer_to_cxi returns early when buffer is empty - so no file.
    assert stats["total_events_written"] == 0
    assert stats["total_events_filtered"] == 1
    assert stats["chunks_written"] == 0

    cxi_files = _list_cxi_files(tmp_path)
    assert cxi_files == [], f"expected no .cxi files, got {cxi_files}"


# -----------------------------------------------------------------------------
# (d) peak coordinates written to CXI match ground truth
# -----------------------------------------------------------------------------

def test_peak_coords_match_ground_truth(ray_cluster, tmp_path):
    """peakYPosRaw / peakXPosRaw recover injected peak centers within 1px."""
    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    B, C, H, W = 1, 4, 64, 64
    peaks_per_panel = 3

    pipeline_output, ground_truth = _make_pipeline_output(
        B=B,
        C=C,
        H=H,
        W=W,
        peaks_per_panel=peaks_per_panel,
        seed=0,
    )
    process_batch(pipeline_output, writer)
    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1

    with h5py.File(cxi_files[0], "r") as f:
        n_peaks = f["/entry_1/result_1/nPeaks"][()]
        peak_y = f["/entry_1/result_1/peakYPosRaw"][()]
        peak_x = f["/entry_1/result_1/peakXPosRaw"][()]

    # One event in this CXI, all C panels merged.
    assert n_peaks[0] == peaks_per_panel * C

    # Ground truth for event 0 = union of peaks across all C panels.
    gt_event_0 = []
    for panel_gt in ground_truth:
        gt_event_0.extend(panel_gt)

    written_y = peak_y[0, : int(n_peaks[0])]
    written_x = peak_x[0, : int(n_peaks[0])]
    _greedy_match_peaks(written_y, written_x, gt_event_0, tol=1.0)


# -----------------------------------------------------------------------------
# (e) nPeaks per event matches the number of ground-truth peaks
# -----------------------------------------------------------------------------

def test_npeaks_matches(ray_cluster, tmp_path):
    """nPeaks[i] equals the total ground-truth peaks across C panels of event i."""
    writer = _make_writer(tmp_path, buffer_size=2, min_num_peak=1)

    # Two batches, each B=1, C=4 with different peaks_per_panel.
    peaks_per_panel_by_batch = [3, 5]

    expected_counts: List[int] = []
    for batch_idx, ppp in enumerate(peaks_per_panel_by_batch):
        pipeline_output, ground_truth = _make_pipeline_output(
            B=1, C=4, H=64, W=64, peaks_per_panel=ppp, seed=batch_idx,
        )
        process_batch(pipeline_output, writer)
        # One event per batch; all C panels' peaks merge into that event.
        expected_counts.append(sum(len(p) for p in ground_truth))

    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1

    with h5py.File(cxi_files[0], "r") as f:
        n_peaks = f["/entry_1/result_1/nPeaks"][()]

    assert n_peaks.tolist() == expected_counts, (
        f"nPeaks {n_peaks.tolist()} != expected {expected_counts}"
    )
