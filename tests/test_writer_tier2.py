"""
Tier 2 end-to-end writer tests for cxi_pipeline_ray.

Builds on Tier 1 (tests/test_writer_offline.py) by exercising:

  * the detector-image reconstruction path (coordinator.py:160 ->
    reconstruction.reconstruct_from_arrays -> file_writer.py:_flush_buffer_to_cxi
    /entry_1/data_1/data dataset),
  * physics metadata flow (photon_wavelength -> photon_energy_eV via
    cxi_pipeline_ray.core.reconstruction.wavelength_to_energy, timestamp
    roundtrip in /LCLS/detector_1/timestamp),
  * peak coordinate clipping (coordinator.py: drop peaks with y >= H_orig
    or x >= W_orig under bottom-right padding).

Uses make_synthetic_pipeline_output from tests/fixtures.py (Task 17), which
wires up the ray.put detector image, FakePreprocessingMetadata, and
metadata dict in one call.

Image shape note (file_writer.py:194 in submit_processed_batch):

    if len(img.shape) == 3:           # (C, H_orig, W_orig)
        cheetah_image = img.reshape(-1, img.shape[-1])  # -> (C*H_orig, W_orig)

So the per-event image written to /entry_1/data_1/data has shape
(C * H_orig, W_orig). Tests assert this exact shape - no guessing.
"""

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
from cxi_pipeline_ray.core.reconstruction import wavelength_to_energy
from tests.fixtures import make_synthetic_pipeline_output


# -----------------------------------------------------------------------------
# Ray lifecycle - module-scoped so all tests share one local cluster.
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ray_cluster():
    """
    Start a small local Ray cluster for the duration of this module.

    `_temp_dir` is deliberately rooted at /tmp with a short prefix because
    Ray creates a plasma store AF_UNIX socket inside it and Linux caps
    socket paths at 108 bytes.
    """
    tmp = tempfile.mkdtemp(prefix="rt2-", dir="/tmp")
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

def _make_writer(
    tmp_path,
    buffer_size: int = 1,
    min_num_peak: int = 1,
    max_num_peak: int = 1024,
    crystfel_mode: bool = False,
    save_segmentation_maps: bool = False,
):
    return CXIFileWriterActor.remote(
        output_dir=str(tmp_path),
        geom_file=None,
        buffer_size=buffer_size,
        min_num_peak=min_num_peak,
        max_num_peak=max_num_peak,
        file_prefix="tier2",
        crystfel_mode=crystfel_mode,
        save_segmentation_maps=save_segmentation_maps,
    )


def _list_cxi_files(tmp_path) -> List[str]:
    return sorted(glob.glob(os.path.join(str(tmp_path), "*.cxi")))


def _greedy_match_peaks(
    recovered_y: np.ndarray,
    recovered_x: np.ndarray,
    ground_truth: List[Tuple[int, int]],
    tol: float = 1.0,
) -> None:
    """Greedy nearest-neighbor match; asserts every recovered peak finds a
    ground-truth peak within `tol` and every ground-truth peak is matched."""
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
            f"{best_dist:.3f}px from nearest ground-truth (tolerance={tol})"
        )
        used.add(best_gt)


# -----------------------------------------------------------------------------
# (a) detector image survives: /entry_1/data_1/data is the assembled (C*H, W)
# -----------------------------------------------------------------------------

def test_detector_image_in_cxi(ray_cluster, tmp_path):
    """
    Feed a fully populated PipelineOutput; assert the writer wrote
    /entry_1/data_1/data with the documented vertically-stacked shape
    (B, C*H_orig, W_orig) and dtype float32.

    Uses 1667/1668/1696/1696 to mirror the production ePix10k2M shapes
    called out in the Task 18 spec, but B=1 C=4 keeps runtime well under a
    minute (per-event image ~26 MB).
    """
    B, C = 1, 4
    H_orig, W_orig = 1667, 1668
    H_preprocessed, W_preprocessed = 1696, 1696

    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    fake_output, _gt = make_synthetic_pipeline_output(
        B=B,
        C=C,
        H_orig=H_orig,
        W_orig=W_orig,
        H_preprocessed=H_preprocessed,
        W_preprocessed=W_preprocessed,
        peaks_per_panel=3,
        photon_wavelength=1.3,
        timestamp=0,
        seed=0,
    )
    process_batch(fake_output, writer)
    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1, f"expected 1 .cxi file, got {len(cxi_files)}"

    with h5py.File(cxi_files[0], "r") as f:
        data = f["/entry_1/data_1/data"]
        # group_panels_into_events emits (C, H_orig, W_orig) per event.
        # file_writer.submit_processed_batch (line 194) reshapes 3D images
        # to (C*H_orig, W_orig) before buffering. So the dataset shape is
        # (num_events=B, C*H_orig, W_orig).
        assert data.shape == (B, C * H_orig, W_orig), (
            f"unexpected detector image shape: {data.shape} "
            f"(expected ({B}, {C * H_orig}, {W_orig}))"
        )
        assert data.dtype == np.float32, f"expected float32, got {data.dtype}"


# -----------------------------------------------------------------------------
# (b) photon_wavelength -> photon_energy_eV via wavelength_to_energy
# -----------------------------------------------------------------------------

def test_photon_energy_from_wavelength(ray_cluster, tmp_path):
    """
    Confirm coordinator.process_batch converts photon_wavelength into
    photon_energy and that the writer stores it under /LCLS/photon_energy_eV.
    """
    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    fake_output, _gt = make_synthetic_pipeline_output(
        B=1, C=4,
        H_orig=64, W_orig=64,
        H_preprocessed=80, W_preprocessed=80,
        peaks_per_panel=3,
        photon_wavelength=1.3,
        timestamp=0,
        seed=0,
    )
    process_batch(fake_output, writer)
    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1

    expected_eV = wavelength_to_energy(1.3)
    with h5py.File(cxi_files[0], "r") as f:
        photon_energies = f["/LCLS/photon_energy_eV"][()]

    assert photon_energies.shape == (1,)
    assert photon_energies[0] == pytest.approx(expected_eV, rel=1e-5), (
        f"photon_energy_eV {photon_energies[0]} != expected "
        f"{expected_eV} for wavelength=1.3 A"
    )


# -----------------------------------------------------------------------------
# (c) timestamp roundtrip - dataset only emitted when nonzero
# -----------------------------------------------------------------------------

def test_timestamp_roundtrip(ray_cluster, tmp_path):
    """
    Use a uint64 timestamp with seconds=1 in the upper 32 bits and
    nanoseconds=2 in the lower 32 bits (matches psana2 convention noted in
    file_writer.py and reconstruction.py). Assert /LCLS/detector_1/timestamp
    exists and round-trips the exact value.
    """
    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    ts = (1 << 32) | 2

    fake_output, _gt = make_synthetic_pipeline_output(
        B=1, C=4,
        H_orig=64, W_orig=64,
        H_preprocessed=80, W_preprocessed=80,
        peaks_per_panel=3,
        photon_wavelength=1.3,
        timestamp=ts,
        seed=0,
    )
    process_batch(fake_output, writer)
    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1

    with h5py.File(cxi_files[0], "r") as f:
        # Per file_writer.py the timestamp dataset is only created when
        # `np.any(timestamps_array != 0)`; a nonzero ts must yield it.
        assert "/LCLS/detector_1/timestamp" in f, (
            "timestamp dataset missing despite nonzero timestamp"
        )
        ts_dataset = f["/LCLS/detector_1/timestamp"][()]
        assert ts_dataset.dtype == np.uint64, (
            f"expected uint64, got {ts_dataset.dtype}"
        )
        assert ts_dataset.shape == (1,)
        assert int(ts_dataset[0]) == ts, (
            f"timestamp roundtrip mismatch: {int(ts_dataset[0])} != {ts}"
        )


# -----------------------------------------------------------------------------
# (d) Peak placed in padded region (>= H_orig or >= W_orig) is clipped.
# -----------------------------------------------------------------------------

def test_edge_peak_clipped(ray_cluster, tmp_path):
    """
    Place ground-truth peaks beyond the original detector bounds (in the
    bottom-right padded region). coordinator.process_batch must drop them
    before the writer sees them, so they never appear in the CXI.
    """
    B, C = 1, 4
    H_orig, W_orig = 64, 64
    H_preprocessed, W_preprocessed = 80, 80

    # One panel gets a peak right at (H_preprocessed-7, W_preprocessed-7) so
    # the entire blob fits inside the panel but the peak center is OUTSIDE
    # the original-detector window. Other panels also get out-of-bounds
    # peaks, so after clipping every panel should report zero peaks and
    # min_num_peak=1 will filter the event entirely.
    out_y = H_preprocessed - 7
    out_x = W_preprocessed - 7
    peak_locations = [[(out_y, out_x)] for _ in range(B * C)]

    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    fake_output, gt_in_orig = make_synthetic_pipeline_output(
        B=B, C=C,
        H_orig=H_orig, W_orig=W_orig,
        H_preprocessed=H_preprocessed, W_preprocessed=W_preprocessed,
        peak_locations=peak_locations,
        photon_wavelength=1.3,
        timestamp=0,
        seed=0,
    )

    # Sanity: the fixture's clipped ground truth should be empty for all
    # panels (every peak was placed outside the original window).
    assert all(len(panel) == 0 for panel in gt_in_orig), (
        f"expected all panels to clip empty, got {gt_in_orig}"
    )

    process_batch(fake_output, writer)
    stats = ray.get(writer.flush_final.remote())

    # Event has zero peaks after clipping; min_num_peak=1 filters it out
    # before it lands in the buffer, so no CXI file is written.
    assert stats["total_events_written"] == 0
    assert stats["total_events_filtered"] == 1
    assert stats["chunks_written"] == 0

    cxi_files = _list_cxi_files(tmp_path)
    assert cxi_files == [], f"expected no .cxi files, got {cxi_files}"


# -----------------------------------------------------------------------------
# (e) Peak inside original bounds survives clipping and lands in the CXI.
# -----------------------------------------------------------------------------

def test_edge_peak_within_orig(ray_cluster, tmp_path):
    """
    Peak placed at (H_orig-2, W_orig-2) is strictly < (H_orig, W_orig) and
    must survive the bottom-right clip in coordinator.process_batch.
    """
    B, C = 1, 4
    H_orig, W_orig = 64, 64
    H_preprocessed, W_preprocessed = 80, 80

    in_y = H_orig - 2
    in_x = W_orig - 2

    # Each panel gets exactly one peak inside the original bounds.
    peak_locations = [[(in_y, in_x)] for _ in range(B * C)]

    writer = _make_writer(tmp_path, buffer_size=1, min_num_peak=1)

    fake_output, gt_in_orig = make_synthetic_pipeline_output(
        B=B, C=C,
        H_orig=H_orig, W_orig=W_orig,
        H_preprocessed=H_preprocessed, W_preprocessed=W_preprocessed,
        peak_locations=peak_locations,
        photon_wavelength=1.3,
        timestamp=0,
        seed=0,
    )

    # All four panels keep their single peak through clipping.
    assert all(len(panel) == 1 for panel in gt_in_orig), (
        f"expected one peak per panel after clipping, got {gt_in_orig}"
    )

    process_batch(fake_output, writer)
    ray.get(writer.flush_final.remote())

    cxi_files = _list_cxi_files(tmp_path)
    assert len(cxi_files) == 1, f"expected 1 .cxi file, got {len(cxi_files)}"

    with h5py.File(cxi_files[0], "r") as f:
        n_peaks = f["/entry_1/result_1/nPeaks"][()]
        peak_y = f["/entry_1/result_1/peakYPosRaw"][()]
        peak_x = f["/entry_1/result_1/peakXPosRaw"][()]

    # One event in this CXI; all C panels merged - so C peaks total.
    assert n_peaks[0] == C, (
        f"expected {C} peaks (one per panel), got {n_peaks[0]}"
    )

    written_y = peak_y[0, : int(n_peaks[0])]
    written_x = peak_x[0, : int(n_peaks[0])]

    # Every recovered peak must match (in_y, in_x) within 1px.
    expected_event = [(in_y, in_x) for _ in range(C)]
    _greedy_match_peaks(written_y, written_x, expected_event, tol=1.0)


# -----------------------------------------------------------------------------
# (f) crystfel_mode requires a .geom fixture - deferred to follow-up work.
# -----------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="crystfel_mode requires a real CrystFEL .geom fixture - follow-up work",
    strict=False,
    run=False,
)
def test_crystfel_mode_requires_geom(ray_cluster, tmp_path):
    """
    Placeholder for a future test that supplies a minimal .geom file and
    asserts crystfel_mode-specific datasets (peakTotalIntensity, EncoderValue
    populated from metadata, etc.) are emitted. Skipped until a fixture .geom
    is added.
    """
    raise NotImplementedError(
        "Need a minimal CrystFEL .geom fixture to exercise crystfel_mode."
    )
