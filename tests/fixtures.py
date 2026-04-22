"""
Test fixtures for cxi-pipeline-ray.

Provides deterministic, GPU-free inputs for Tier 1/2/3 tests:

- make_synthetic_logits(B, C, H, W, ...): builds (B*C, 2, H, W) float32 logits
  that argmax to known peak locations. Pairs with ground truth returned for
  assertion.
- FakePipelineOutput: duck-typed stand-in for what
  cxi_pipeline_ray.core.coordinator.process_batch reads from a real
  PipelineOutput. torch is imported lazily inside get_torch_tensor; if torch
  is unavailable in the venv, a numpy-shim with .numpy() and .shape is
  returned instead so coordinator.process_batch continues to work.

Tier 2/3 builders (full PipelineOutput with detector images + physics
metadata) are added in a later task.
"""

from typing import List, Optional, Tuple

import numpy as np


def make_synthetic_logits(
    B: int,
    C: int,
    H: int,
    W: int,
    peaks_per_panel: int = 3,
    peak_locations: Optional[List[List[Tuple[int, int]]]] = None,
    peak_blob_size: int = 3,
    seed: int = 0,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
    """
    Build a (B*C, 2, H, W) float32 logits tensor that argmaxes to known peaks.

    Class convention (matches cxi_pipeline_ray.core.peak_finding):
        class 0 = background (argmax here -> no peak)
        class 1 = peak (argmax here -> peak)

    Algorithm:
        - Initialize the whole tensor with class-0 logit = +10.0 and class-1
          logit = -10.0 (argmax = 0 everywhere -> no peaks anywhere).
        - For each peak at integer (y, x), overwrite a square region
          [y - peak_blob_size : y + peak_blob_size + 1,
           x - peak_blob_size : x + peak_blob_size + 1]
          with class-1 logit = +20.0 and class-0 logit = -10.0
          (argmax = 1 there -> a peak blob).

    Args:
        B: Number of events in the batch.
        C: Number of panels per event.
        H: Panel height in pixels.
        W: Panel width in pixels.
        peaks_per_panel: Number of peaks per panel when sampling (ignored if
            peak_locations is provided).
        peak_locations: If provided, list of length B*C where each entry is a
            list of (y, x) integer tuples naming peak centers for that panel.
            Pass empty lists to get empty panels.
        peak_blob_size: Half-side of each peak blob. The blob occupies a
            (2*peak_blob_size + 1) x (2*peak_blob_size + 1) square centered at
            the peak coordinate.
        seed: PRNG seed used when sampling peak locations.

    Returns:
        (logits, ground_truth) where:
            logits: np.ndarray, shape (B*C, 2, H, W), dtype float32.
            ground_truth: list of length B*C; each entry is a list of (y, x)
                integer tuples naming the blob centers on that panel.
    """
    num_panels = B * C

    # Initialize argmax = 0 everywhere (background).
    logits = np.empty((num_panels, 2, H, W), dtype=np.float32)
    logits[:, 0, :, :] = 10.0
    logits[:, 1, :, :] = -10.0

    # Either use caller-provided peak_locations or sample new ones.
    if peak_locations is not None:
        if len(peak_locations) != num_panels:
            raise ValueError(
                f"peak_locations must have length B*C = {num_panels}, "
                f"got {len(peak_locations)}"
            )
        ground_truth = [list(panel) for panel in peak_locations]
    else:
        ground_truth = _sample_peak_locations(
            num_panels=num_panels,
            H=H,
            W=W,
            peaks_per_panel=peaks_per_panel,
            peak_blob_size=peak_blob_size,
            seed=seed,
        )

    # Paint each peak blob onto its panel.
    for panel_idx, panel_peaks in enumerate(ground_truth):
        for (y, x) in panel_peaks:
            y0 = max(0, y - peak_blob_size)
            y1 = min(H, y + peak_blob_size + 1)
            x0 = max(0, x - peak_blob_size)
            x1 = min(W, x + peak_blob_size + 1)
            logits[panel_idx, 0, y0:y1, x0:x1] = -10.0
            logits[panel_idx, 1, y0:y1, x0:x1] = 20.0

    return logits, ground_truth


def _sample_peak_locations(
    num_panels: int,
    H: int,
    W: int,
    peaks_per_panel: int,
    peak_blob_size: int,
    seed: int,
) -> List[List[Tuple[int, int]]]:
    """
    Sample `peaks_per_panel` integer (y, x) coordinates per panel with a
    minimum inter-peak separation of 2*peak_blob_size + 2 pixels (so that
    scipy.ndimage.label under 8-connectivity cannot merge two peaks into a
    single component).

    Returns a list of length num_panels; each entry is a list of (y, x)
    tuples.
    """
    rng = np.random.default_rng(seed)

    # Stay strictly inside the panel so the full blob fits.
    y_min = peak_blob_size + 1
    y_max = H - peak_blob_size - 1
    x_min = peak_blob_size + 1
    x_max = W - peak_blob_size - 1

    min_separation = 2 * peak_blob_size + 2

    if y_max <= y_min or x_max <= x_min:
        raise ValueError(
            f"Panel too small ({H}x{W}) for peak_blob_size={peak_blob_size}"
        )

    ground_truth: List[List[Tuple[int, int]]] = []

    for _ in range(num_panels):
        chosen: List[Tuple[int, int]] = []
        attempts = 0
        max_attempts = peaks_per_panel * 200  # Generous budget for rejection.

        while len(chosen) < peaks_per_panel and attempts < max_attempts:
            attempts += 1
            y = int(rng.integers(y_min, y_max))
            x = int(rng.integers(x_min, x_max))

            too_close = False
            for (yy, xx) in chosen:
                if abs(yy - y) < min_separation and abs(xx - x) < min_separation:
                    too_close = True
                    break

            if not too_close:
                chosen.append((y, x))

        if len(chosen) < peaks_per_panel:
            raise RuntimeError(
                f"Could not place {peaks_per_panel} peaks in a {H}x{W} panel "
                f"with separation {min_separation}; reduce peaks_per_panel "
                f"or enlarge the panel."
            )

        ground_truth.append(chosen)

    return ground_truth


class _NumpyTensorShim:
    """
    Minimal stand-in for a torch tensor when torch is not installed.

    Exposes just enough of the torch tensor surface for
    coordinator.process_batch, which calls
    ``pipeline_output.get_torch_tensor(device='cpu').numpy()``.
    """

    def __init__(self, array: np.ndarray):
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array

    @property
    def shape(self):
        return self._array.shape

    def __array__(self):
        return self._array


class FakePipelineOutput:
    """
    Duck-typed stand-in for a real PipelineOutput as consumed by
    cxi_pipeline_ray.core.coordinator.process_batch.

    Only the attributes the coordinator reads are populated. Defaults match
    the "no metadata / no detector image" path the coordinator warns about
    but still handles.

    Attributes:
        preprocessing_metadata: Object exposing `.original_shape` and
            `.preprocessed_shape` tuples, or None.
        original_image_ref: A ray ObjectRef to the detector image tensor, or
            None.
        metadata: Dict of physics metadata (photon_wavelength, timestamp).
    """

    def __init__(
        self,
        logits: np.ndarray,
        preprocessing_metadata=None,
        original_image_ref=None,
        metadata: Optional[dict] = None,
    ):
        self._logits = logits
        self.preprocessing_metadata = preprocessing_metadata
        self.original_image_ref = original_image_ref
        self.metadata = metadata if metadata is not None else {}

    def get_torch_tensor(self, device: str = "cpu"):
        """
        Return a torch tensor wrapping the stored logits.

        torch is imported lazily — if it is not installed in the venv (as is
        the case for the cxi-pipeline-ray dev env), a numpy-shim object is
        returned instead so downstream code that only calls `.numpy()` on the
        result still works.
        """
        try:
            import torch  # noqa: WPS433 (intentional lazy import)

            tensor = torch.from_numpy(self._logits)
            if device != "cpu":
                tensor = tensor.to(device)
            return tensor
        except ImportError:
            return _NumpyTensorShim(self._logits)
