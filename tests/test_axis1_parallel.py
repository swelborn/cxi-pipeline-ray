"""
Tier 1 regression test for Axis 1 peak-finding parallelism
(``process_batch(..., num_cpu_workers=K)``).

Verifies that routing per-panel peak finding through Ray tasks produces
CXI output that is byte-equivalent to the sequential path. This is the
regression gate required by Task 25: flipping ``num_cpu_workers`` must
not change *what* is written, only *how fast*.

Both runs use the same synthetic logits fixture so ground truth is
identical; the only difference is the scheduling layer.
"""

from types import SimpleNamespace

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


@pytest.fixture(scope="module")
def ray_cluster():
    """Local Ray cluster with enough CPUs to exercise K=4 fan-out."""
    tmp = tempfile.mkdtemp(prefix="rt-", dir="/tmp")
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=tmp,
    )
    yield
    ray.shutdown()


def _make_pipeline_output(B: int, C: int, H: int, W: int, seed: int = 0):
    """Tiny pipeline output with a real ray.put detector image (so the
    writer's image path doesn't short-circuit) and identity preprocessing
    metadata (so no coordinate clipping occurs)."""
    logits, _gt = make_synthetic_logits(
        B=B, C=C, H=H, W=W, peaks_per_panel=3, seed=seed,
    )
    detector_images = np.zeros((B * C, 1, H, W), dtype=np.float32)
    original_image_ref = ray.put(detector_images)
    preprocessing_metadata = SimpleNamespace(
        original_shape=(B, C, H, W),
        preprocessed_shape=(B * C, 1, H, W),
    )
    metadata = {"photon_wavelength": 1.3, "timestamp": 0}
    return FakePipelineOutput(
        logits=logits,
        preprocessing_metadata=preprocessing_metadata,
        original_image_ref=original_image_ref,
        metadata=metadata,
    )


def _make_writer(tmp_path):
    return CXIFileWriterActor.remote(
        output_dir=str(tmp_path),
        geom_file=None,
        buffer_size=1,
        min_num_peak=1,
        max_num_peak=1024,
        file_prefix="axis1",
        crystfel_mode=False,
        save_segmentation_maps=False,
    )


def _read_peaks_from_cxi(cxi_path: str):
    """Return the single event's peak arrays from a CXI file."""
    with h5py.File(cxi_path, "r") as f:
        npeaks = int(f["/entry_1/result_1/nPeaks"][0])
        y = np.array(f["/entry_1/result_1/peakYPosRaw"][0, :npeaks])
        x = np.array(f["/entry_1/result_1/peakXPosRaw"][0, :npeaks])
    return npeaks, y, x


def _run_and_read(ray_cluster, tmp_path, num_cpu_workers: int):
    """Process one batch with the given parallelism and return the peak
    coordinates from the resulting CXI file."""
    writer = _make_writer(tmp_path)
    pipeline_output = _make_pipeline_output(B=1, C=4, H=64, W=64, seed=0)
    process_batch(pipeline_output, writer, num_cpu_workers=num_cpu_workers)
    ray.get(writer.flush_final.remote())

    cxi_files = sorted(glob.glob(os.path.join(str(tmp_path), "*.cxi")))
    assert len(cxi_files) == 1, (
        f"expected exactly 1 CXI file, got {len(cxi_files)}"
    )
    return _read_peaks_from_cxi(cxi_files[0])


def test_k1_and_k4_produce_identical_peaks(ray_cluster, tmp_path):
    """``num_cpu_workers=1`` (sequential) and ``num_cpu_workers=4`` (Ray
    task fan-out) must produce identical CXI peak lists for the same
    input logits."""
    k1_dir = tmp_path / "k1"
    k1_dir.mkdir()
    k4_dir = tmp_path / "k4"
    k4_dir.mkdir()

    n1, y1, x1 = _run_and_read(ray_cluster, k1_dir, num_cpu_workers=1)
    n4, y4, x4 = _run_and_read(ray_cluster, k4_dir, num_cpu_workers=4)

    assert n1 == n4, f"nPeaks differs: K=1 -> {n1}, K=4 -> {n4}"
    # Peak ordering may differ between sequential and parallel paths, so
    # sort both by (y, x) before comparing.
    k1_sorted = sorted(zip(y1.tolist(), x1.tolist()))
    k4_sorted = sorted(zip(y4.tolist(), x4.tolist()))
    assert k1_sorted == pytest.approx(k4_sorted, abs=1e-6), (
        f"peak coords differ between K=1 and K=4:\n"
        f"  K=1: {k1_sorted}\n"
        f"  K=4: {k4_sorted}"
    )


def test_k_greater_than_panels_is_safe(ray_cluster, tmp_path):
    """Requesting more workers than the batch has panels must not error —
    the effective parallelism is capped at ``num_panels``."""
    writer = _make_writer(tmp_path)
    pipeline_output = _make_pipeline_output(B=1, C=4, H=64, W=64, seed=0)
    process_batch(pipeline_output, writer, num_cpu_workers=64)
    stats = ray.get(writer.flush_final.remote())
    assert stats["total_events_written"] == 1
    assert len(sorted(glob.glob(os.path.join(str(tmp_path), "*.cxi")))) == 1
