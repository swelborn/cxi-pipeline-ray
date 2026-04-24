"""
Synchronous pipeline coordinator for CXI post-processing.

Pulls batches from Q2, runs peak finding, groups panels into events,
and submits results to the CXI file writer actor.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import ray

from .peak_finding import find_peaks_numpy
from .reconstruction import reconstruct_from_arrays, wavelength_to_energy


def _find_peaks_panel(
    panel_logits: np.ndarray,
    H_orig: Optional[int],
    W_orig: Optional[int],
    save_segmentation_maps: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run peak finding on a single panel's logits and clip peaks to original
    detector bounds.

    This is the per-panel body that used to live inline in `process_batch`.
    It is pulled out so it can be called both sequentially (num_cpu_workers=1)
    and as the body of a Ray remote task (num_cpu_workers>1) without logic
    drift between the two code paths.

    Args:
        panel_logits: (2, H, W) logits for one panel.
        H_orig: Original detector height before bottom-right padding, or None
            to skip the clip.
        W_orig: Original detector width before bottom-right padding, or None
            to skip the clip.
        save_segmentation_maps: If True, also return the (H_orig, W_orig)
            uint8 segmentation map (clipped to original bounds).

    Returns:
        (peaks, seg_map) where
        - peaks is an (N, 3) float array with rows [panel_idx=0, y, x], peaks
          outside (H_orig, W_orig) already dropped.
        - seg_map is None when save_segmentation_maps=False, otherwise a
          uint8 array clipped to (H_orig, W_orig) when those are provided.
    """
    if save_segmentation_maps:
        peaks, seg_map = find_peaks_numpy(panel_logits, return_seg_map=True)
    else:
        peaks = find_peaks_numpy(panel_logits)
        seg_map = None

    # Clip peaks to original bounds (bottom-right padding assumption).
    if H_orig is not None and W_orig is not None:
        peaks_clipped = []
        for peak in peaks:
            _, y, x = peak
            if y < H_orig and x < W_orig:
                peaks_clipped.append([0, y, x])

        if peaks_clipped:
            peaks = np.array(peaks_clipped)
        else:
            peaks = np.array([]).reshape(0, 3)

        if save_segmentation_maps and seg_map is not None:
            seg_map = seg_map[:H_orig, :W_orig]

    return peaks, seg_map


@ray.remote(num_cpus=1)
def _find_peaks_task(
    panel_logits: np.ndarray,
    H_orig: Optional[int],
    W_orig: Optional[int],
    save_segmentation_maps: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Ray remote wrapper around `_find_peaks_panel`. One task per panel.

    Runs on any CPU in the Ray cluster; the per-panel ndarray is the only
    input that crosses the Ray object store, so serialization cost scales
    with panel size, not with batch size.

    Each task declares ``num_cpus=1`` as a hard Ray resource requirement, so
    the cluster must provide at least ``num_panels`` CPUs to reach the
    fan-out ceiling — with fewer, Ray queues the excess tasks.
    """
    return _find_peaks_panel(panel_logits, H_orig, W_orig, save_segmentation_maps)


def group_panels_into_events(batch_info):
    """
    Group panels into events for CXI file writing.

    Args:
        batch_info: Dictionary with keys:
            - completed_panels: List of (panel_idx, peaks_list, image)
            - B, C: Batch size and panels per event
            - H_orig, W_orig: Original detector dimensions
            - detector_images_4d: (B, C, H, W) detector images
            - photon_energy, timestamp, photon_wavelength: Physics metadata
            - segmentation_maps: Optional list of (H, W) seg_maps (debug mode)

    Returns:
        (event_images, event_peaks, event_metadata, event_seg_maps): Tuples for file_writer.submit_processed_batch()
        Note: event_seg_maps can be None if segmentation_maps not provided
    """
    B = batch_info["B"]
    C = batch_info["C"]
    H_orig = batch_info["H_orig"]
    W_orig = batch_info["W_orig"]
    detector_images_4d = batch_info["detector_images_4d"]
    completed_panels = batch_info["completed_panels"]
    segmentation_maps = batch_info.get("segmentation_maps", None)

    # Sort by panel index
    completed_panels.sort(key=lambda x: x[0])

    # Group into events
    event_images = []
    event_peaks = []
    event_metadata = []
    event_seg_maps = [] if segmentation_maps is not None else None

    for event_idx in range(B):
        panel_start = event_idx * C
        panel_end = (event_idx + 1) * C

        # Get detector image for this event: (C, H, W)
        if detector_images_4d is not None:
            event_image = detector_images_4d[event_idx]  # (C, H_orig, W_orig)
        else:
            # Fallback: use logits (won't be perfect but better than nothing)
            event_image = None

        # Get segmentation maps for this event: (C, H, W)
        if segmentation_maps is not None:
            event_seg_map = np.stack(
                segmentation_maps[panel_start:panel_end]
            )  # (C, H_orig, W_orig)
        else:
            event_seg_map = None

        # Combine peaks from all panels for this event
        event_peaks_combined = []
        for panel_idx in range(panel_start, panel_end):
            # Find peaks for this panel
            panel_data = next((p for p in completed_panels if p[0] == panel_idx), None)
            if panel_data is None:
                continue

            _, panel_peaks, _ = panel_data

            # Add actual panel index (relative to event) to each peak
            for peak in panel_peaks:
                _, y, x = peak
                # Clip to original detector bounds (bottom-right padding)
                if H_orig and W_orig and (y >= H_orig or x >= W_orig):
                    continue  # Skip peaks outside original detector area
                event_peaks_combined.append([int(panel_idx % C), float(y), float(x)])

        # Create metadata for this event
        photon_energy = batch_info["photon_energy"]
        photon_wavelength = batch_info["photon_wavelength"]
        timestamp = batch_info["timestamp"]

        # Handle array metadata (take event_idx element)
        if isinstance(photon_energy, (list, np.ndarray)):
            photon_energy = (
                float(photon_energy[event_idx])
                if len(photon_energy) > event_idx
                else float(photon_energy[0])
            )
        elif photon_energy is not None:
            photon_energy = float(photon_energy)

        if isinstance(timestamp, (list, np.ndarray)):
            timestamp = (
                int(timestamp[event_idx]) if len(timestamp) > event_idx else int(timestamp[0])
            )
        elif timestamp is not None:
            timestamp = int(timestamp)

        event_meta = {
            "photon_energy": photon_energy if photon_energy is not None else 0.0,
            "timestamp": timestamp if timestamp is not None else 0,
        }

        if photon_wavelength is not None:
            if isinstance(photon_wavelength, (list, np.ndarray)):
                event_meta["photon_wavelength"] = (
                    float(photon_wavelength[event_idx])
                    if len(photon_wavelength) > event_idx
                    else float(photon_wavelength[0])
                )
            else:
                event_meta["photon_wavelength"] = float(photon_wavelength)

        event_images.append(event_image)
        event_peaks.append(event_peaks_combined)
        event_metadata.append(event_meta)
        if event_seg_maps is not None:
            event_seg_maps.append(event_seg_map)

    return event_images, event_peaks, event_metadata, event_seg_maps


def process_batch(
    pipeline_output,
    file_writer,
    save_segmentation_maps: bool = False,
    num_cpu_workers: int = 1,
):
    """
    Process a single batch from Q2 through peak finding and submit to writer.

    Peak finding is done panel-by-panel. When `num_cpu_workers > 1`, each
    panel runs as a Ray task (Axis 1 parallelism — see
    docs/design/bottleneck-fix-decision.md). When `num_cpu_workers == 1`,
    the sequential path runs in-process so behavior matches the pre-Axis-1
    baseline.

    Args:
        pipeline_output: PipelineOutput object from Q2
        file_writer: CXIFileWriterActor instance
        save_segmentation_maps: Save segmentation maps to CXI (debug mode)
        num_cpu_workers: Number of parallel Ray tasks to use for per-panel
            peak finding. 1 = sequential (no Ray), matches baseline behavior.

    Returns:
        Number of events processed
    """
    # Extract logits
    logits = pipeline_output.get_torch_tensor(device="cpu").numpy()  # (B*C, num_classes, H, W)

    # Extract B, C, H_orig, W_orig from preprocessing metadata
    B, C, H_orig, W_orig = None, None, None, None

    if (
        hasattr(pipeline_output, "preprocessing_metadata")
        and pipeline_output.preprocessing_metadata is not None
    ):
        original_shape = pipeline_output.preprocessing_metadata.original_shape
        B, C, H_orig, W_orig = original_shape
        logging.debug(f"Extracted shape from metadata: B={B}, C={C}, H={H_orig}, W={W_orig}")
    else:
        logging.warning(
            "NO preprocessing_metadata found! This may cause detector image/seg map mismatch!"
        )

    # Try to extract detector images (can fail independently)
    detector_images_4d = None

    if (
        hasattr(pipeline_output, "original_image_ref")
        and pipeline_output.original_image_ref is not None
    ):
        try:
            if (
                hasattr(pipeline_output, "preprocessing_metadata")
                and pipeline_output.preprocessing_metadata is not None
            ):
                ref = pipeline_output.original_image_ref
                try:
                    original_image = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
                except Exception:
                    # ObjectRef owner (actor) has exited; fall back to the direct array copy
                    original_image = getattr(pipeline_output, "original_image", None)
                if original_image is not None:
                    preprocessed_shape = pipeline_output.preprocessing_metadata.preprocessed_shape
                    detector_images_4d = reconstruct_from_arrays(
                        original_image, original_shape, preprocessed_shape
                    )
                    logging.debug(f"Reconstructed detector images: {detector_images_4d.shape}")
            else:
                ref = pipeline_output.original_image_ref
                try:
                    original_image_raw = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
                except Exception:
                    original_image_raw = getattr(pipeline_output, "original_image", None)
                if original_image_raw is not None:
                    logging.warning(
                        f"NO preprocessing metadata - using images as-is: {original_image_raw.shape}"
                    )
                    detector_images_4d = original_image_raw
        except Exception as e:
            logging.warning(f"Failed to extract detector images: {e}")
            detector_images_4d = None
    else:
        logging.warning("NO original_image_ref found! Detector images will be None")

    # Extract physics metadata
    metadata = pipeline_output.metadata if hasattr(pipeline_output, "metadata") else {}

    # Handle photon wavelength → energy conversion
    photon_wavelength = metadata.get("photon_wavelength", None)
    photon_energy = metadata.get("photon_energy", None)

    if photon_wavelength is not None:
        if isinstance(photon_wavelength, (list, np.ndarray)):
            photon_energy = np.array([wavelength_to_energy(w) for w in photon_wavelength])
        else:
            photon_energy = wavelength_to_energy(float(photon_wavelength))

    timestamp = metadata.get("timestamp", None)

    # Run peak finding on logits
    num_panels = logits.shape[0]
    logging.debug(
        f"Running peak finding on {num_panels} panels "
        f"(logits shape: {logits.shape}, num_cpu_workers={num_cpu_workers})..."
    )
    if B and C and num_panels != B * C:
        logging.error(f"MISMATCH: logits.shape[0]={num_panels} but B*C={B*C}!")

    # Axis 1: fan out peak finding across Ray tasks when num_cpu_workers > 1.
    # At K=1 we stay on the sequential path to keep the pre-Axis-1 behavior
    # bit-for-bit identical (no Ray task scheduling, no object-store round
    # trip) — this is what the regression gate in tests/test_writer_*.py
    # relies on.
    if num_cpu_workers > 1 and num_panels > 1:
        refs = [
            _find_peaks_task.remote(
                logits[panel_idx], H_orig, W_orig, save_segmentation_maps
            )
            for panel_idx in range(num_panels)
        ]
        results = ray.get(refs)
    else:
        results = [
            _find_peaks_panel(
                logits[panel_idx], H_orig, W_orig, save_segmentation_maps
            )
            for panel_idx in range(num_panels)
        ]

    all_peaks = [r[0] for r in results]
    all_seg_maps = [r[1] for r in results] if save_segmentation_maps else None

    # Group panels into events
    completed_panels = []
    for panel_idx in range(len(all_peaks)):
        completed_panels.append((panel_idx, all_peaks[panel_idx], None))

    batch_info = {
        "completed_panels": completed_panels,
        "B": B if B is not None else 1,
        "C": C if C is not None else len(all_peaks),
        "H_orig": H_orig,
        "W_orig": W_orig,
        "detector_images_4d": detector_images_4d,
        "photon_energy": photon_energy,
        "photon_wavelength": photon_wavelength,
        "timestamp": timestamp,
        "metadata": metadata,
        "num_panels": len(all_peaks),
        "segmentation_maps": all_seg_maps,
    }

    event_images, event_peaks, event_metadata, event_seg_maps = group_panels_into_events(batch_info)

    # Submit to file writer
    file_writer.submit_processed_batch.remote(
        event_images, event_peaks, event_metadata, event_seg_maps
    )

    total_peaks = sum(len(p) for p in all_peaks)
    logging.debug(f"Submitted {len(event_images)} events, {total_peaks} total peaks")

    return len(event_images)


def run_sync_pipeline(
    q2_manager,
    file_writer,
    batches_per_file: int = 10,
    save_segmentation_maps: bool = False,
    num_cpu_workers: int = 1,
):
    """
    Synchronous pipeline: pull from Q2, process, write CXI files.

    Args:
        q2_manager: ShardedQueueManager for Q2 output queue
        file_writer: CXIFileWriterActor instance (already created)
        batches_per_file: Write CXI file every N batches
        save_segmentation_maps: Save segmentation maps to CXI (debug mode)
        num_cpu_workers: Passed through to process_batch for Axis 1 peak-finding
            parallelism. 1 = sequential (pre-Axis-1 baseline), >1 = Ray tasks.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting synchronous pipeline loop...")
    logger.info(f"Batches per file: {batches_per_file}")
    logger.info(f"Save segmentation maps: {save_segmentation_maps}")
    logger.info(f"Peak-finding parallelism (num_cpu_workers): {num_cpu_workers}")

    # Try to find the StreamingCoordinator by name so we can use its
    # is_completed() method as the exit signal instead of timing heuristics.
    streaming_coordinator = None
    try:
        from peaknet_pipeline_ray.core.coordinator import STREAMING_COORDINATOR_NAME

        streaming_coordinator = ray.get_actor(
            STREAMING_COORDINATOR_NAME, namespace="peaknet-pipeline"
        )
        logger.info("Connected to StreamingCoordinator — will exit when pipeline reports COMPLETED")
    except Exception:
        logger.warning("StreamingCoordinator not found by name — falling back to 60s idle exit")

    batch_count = 0
    total_events = 0
    batches_since_flush = 0
    empty_polls = 0
    IDLE_EXIT_POLLS = 600  # fallback: 600 × 0.1 s = 60 s

    try:
        while True:
            # Pull batch from Q2 (blocking with short timeout)
            pipeline_output = q2_manager.get(timeout=0.1)

            if pipeline_output is None:
                if streaming_coordinator is not None:
                    try:
                        completed = ray.get(streaming_coordinator.is_completed.remote())
                        if completed:
                            q2_size = q2_manager.size()
                            if q2_size == 0:
                                logger.info(
                                    f"StreamingCoordinator reports COMPLETED and Q2 is empty after {batch_count} batches — exiting"
                                )
                                break
                    except Exception:
                        pass  # coordinator exited; fall through to idle check
                if batch_count > 0:
                    empty_polls += 1
                    if empty_polls >= IDLE_EXIT_POLLS:
                        q2_size = q2_manager.size()
                        if q2_size == 0:
                            logger.info(
                                f"Q2 idle for {IDLE_EXIT_POLLS * 0.1:.0f}s and empty after {batch_count} batches — exiting (fallback)"
                            )
                            break
                        empty_polls = 0
                continue

            empty_polls = 0
            q2_size = q2_manager.size()

            # Process batch
            num_events = process_batch(
                pipeline_output,
                file_writer,
                save_segmentation_maps=save_segmentation_maps,
                num_cpu_workers=num_cpu_workers,
            )
            batch_count += 1
            total_events += num_events
            batches_since_flush += 1

            logger.info(
                f"Processed batch {batch_count}: {num_events} events (total: {total_events}) | Q2 remaining: {q2_size}"
            )

            # Periodic flush: write CXI file every N batches
            if batches_since_flush >= batches_per_file:
                logger.info(f"=== Writing CXI file after {batches_since_flush} batches ===")
                stats = ray.get(file_writer.flush_final.remote())
                logger.info(
                    f"Wrote CXI: {stats['chunks_written']} files, "
                    f"{stats['total_events_written']} events written, "
                    f"{stats['total_events_filtered']} events filtered"
                )
                batches_since_flush = 0

            # Progress logging
            if batch_count % 50 == 0:
                logger.info(f"Progress: {batch_count} batches, {total_events} events")

    except KeyboardInterrupt:
        logger.info("\n=== Interrupted by user ===")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Final flush
        logger.info("Flushing final CXI file...")
        stats = ray.get(file_writer.flush_final.remote())

        logger.info("\n=== Processing Complete ===")
        logger.info(f"Total batches: {batch_count}")
        logger.info(f"Total events: {total_events}")
        logger.info(f"Events written: {stats['total_events_written']}")
        logger.info(f"Events filtered: {stats['total_events_filtered']}")
        logger.info(f"CXI files: {stats['chunks_written']}")
