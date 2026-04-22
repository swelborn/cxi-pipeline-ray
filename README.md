# cxi-pipeline-ray

Ray-based CPU post-processing pipeline for PeakNet inference results.

## Overview

`cxi-pipeline-ray` is a scalable, distributed post-processing pipeline that:
- Consumes PeakNet inference results from Ray queues (Q2)
- Performs CPU-based peak finding using scipy.ndimage
- Writes results to CXI files for crystallography analysis

This package implements a hybrid Ray architecture with:
- **Stateless Ray tasks** for parallel peak finding
- **Stateful Ray actor** for buffered CXI file writing
- **Ray best practices** for optimal throughput (backpressure control, batched operations, pipelining)

## Installation

### Development Installation

```bash
cd /sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
pip install -e .
```

### Dependencies

Core dependencies (automatically installed):
- `ray>=2.0.0` - Distributed computing framework
- `numpy>=1.20.0` - Array operations
- `scipy>=1.7.0` - Peak finding algorithms
- `h5py>=3.0.0` - HDF5/CXI file writing
- `torch>=2.0.0` - Tensor operations
- `pyyaml>=6.0` - Configuration loading

External dependencies (must be installed separately):
- `peaknet-pipeline-ray` - For ShardedQueueManager
- `crystfel-stream-parser` - For CheetahConverter (optional)

## Quick Start

### 1. Create Configuration

Copy the example config and customize:

```bash
cp examples/configs/cxi_writer_default.yaml my_config.yaml
# Edit my_config.yaml to set:
#   - ray.namespace (must match pipeline)
#   - queue.name and queue.num_shards (must match pipeline)
#   - output.output_dir
#   - geometry.geom_file (optional)
```

### 2. Run the Writer

```bash
# Basic usage
cxi-writer --config my_config.yaml

# With CLI overrides
cxi-writer --config my_config.yaml \
  --num-cpu-workers 32 \
  --output-dir /custom/output

# Validate config consistency with pipeline
cxi-writer --validate-config \
  --pipeline-config path/to/pipeline_config.yaml \
  --writer-config my_config.yaml
```

### 3. Typical Workflow

```bash
# Terminal 1: Run PeakNet inference pipeline (creates Q2 queue)
peaknet-pipeline --config peaknet-socket-profile-673m-with-output.yaml

# Terminal 2: Start Q2 writer (connects to Q2 and consumes data)
cxi-writer --config my_config.yaml
```

**Note**: While both launch orders work (due to ShardedQueueManager's create-or-connect behavior), launching the pipeline first is recommended as it follows the natural producer→consumer pattern.

## Configuration

See `examples/configs/cxi_writer_default.yaml` for a fully documented configuration file.

### Critical Settings (Must Match Pipeline)

| Setting | Pipeline Location | Writer Location | Why Critical |
|---------|------------------|-----------------|--------------|
| Ray namespace | `ray.namespace` | `ray.namespace` | Must be in same namespace |
| Q2 queue name | `runtime.queue_names.output_queue` | `queue.name` | Must consume from correct queue |
| Q2 num shards | `runtime.queue_num_shards` | `queue.num_shards` | Queue topology must match |

### Key Parameters

- `num_cpu_workers`: Parallel Ray tasks (default: 16)
  - Start with `num_cpu_cores // 4`
  - Increase if CPU utilization is low
  - Decrease if task overhead is high

- `max_pending_tasks`: Backpressure limit (default: 100)
  - Prevents OOM by limiting in-flight work
  - Lower = less memory, higher = better throughput

- `buffer_size`: Events per CXI file (default: 100)
  - Larger = fewer files, more memory

- `min_num_peak`: Minimum peaks to save event (default: 10)
  - Quality filter for events

## Architecture

The pipeline consists of three main components:

1. **Peak Finding (Ray Task)** - `cxi_pipeline_ray/core/peak_finding.py`
   - Stateless CPU-based peak finding
   - Converts logits → segmentation maps → peak positions
   - Uses scipy.ndimage for connected component labeling

2. **File Writer (Ray Actor)** - `cxi_pipeline_ray/core/file_writer.py`
   - Stateful CXI file writer
   - Maintains CheetahConverter for coordinate conversion
   - Buffers events and writes when buffer is full

3. **Coordinator** - `cxi_pipeline_ray/core/coordinator.py`
   - Main pipeline orchestration
   - Implements backpressure control
   - Batched ray.get() calls for optimal throughput
   - Pipelining for overlapping I/O and compute

For detailed architecture documentation, see:
- `PLAN-Q2-CXI-WRITER-v2.md` - Technical design details
- `PLAN-Q2-CXI-WRITER-ARCH.md` - Implementation architecture
- `RAY-BEST-PRACTICES-REVIEW.md` - Ray optimizations

## Testing

The test suite is organized into three tiers that trade speed for realism.
Each tier is independently runnable and exercises a different slice of the
writer output path without needing a running pipeline or a GPU.

| Tier | What it covers | Data source | Ray | GPU | Runtime |
|------|----------------|-------------|-----|-----|---------|
| 1 | Peak-finding correctness and offline writer end-to-end | Hand-crafted logits (argmax to known peaks) | Local cluster for the writer actor only | No | seconds |
| 2 | Detector-image reconstruction, physics metadata, coordinate clipping | Full synthetic `PipelineOutput` (logits + detector image + metadata) | Local cluster | No | ~minute |
| 3 | Q2 injection throughput — load-testing the writer under controlled input rate | Synthetic `PipelineOutput` pushed through `ShardedQueueManager` | Full local cluster + `cxi-writer` subprocess | No | minutes to hours |

Tier 1 and Tier 2 are pytest tests. Tier 3 is a standalone CLI script meant
for before/after comparisons when changing the writer (e.g. Axis-1 or Axis-2
parallelization work).

### How to run

Install the package with dev extras, then run the pytest suite (Tier 1 + 2)
and the benchmark harness (Tier 3) separately:

```bash
# Install pytest, ruff, black alongside the package
pip install -e '.[dev]'

# Tier 1 + Tier 2 (pytest)
pytest tests/ -v

# Tier 3 (standalone harness — see --help for all flags)
python tests/bench_q2_inject.py --help
```

Tier 3 requires `peaknet-pipeline-ray` for its `ShardedQueueManager`:

```bash
pip install git+https://github.com/carbonscott/peaknet-pipeline-ray
```

### Test files

| File | Tier | What it verifies |
|------|------|------------------|
| `tests/fixtures.py` | 1 + 2 | Fixture builders — no assertions, imported by the other test files |
| `tests/test_peak_finding.py` | 1 | `find_peaks_numpy` recovers ground-truth peaks from synthetic logits |
| `tests/test_writer_offline.py` | 1 | `CXIFileWriterActor.process_batch` -> flush -> CXI file write path, using logits-only fixture |
| `tests/test_writer_tier2.py` | 2 | Detector image in `/entry_1/data_1/data`, photon-energy derivation, timestamp round-trip, edge-peak clipping |
| `tests/bench_q2_inject.py` | 3 | Q2 injection throughput — controlled rate, subprocess writer, full report |

### Fixture API

`tests/fixtures.py` exports three builders used across Tier 1 and Tier 2:

- `make_synthetic_logits(B, C, H, W, peaks_per_panel=3, peak_locations=None, peak_blob_size=3, seed=0) -> (logits, ground_truth)`
  — returns a `(B*C, 2, H, W)` float32 logits tensor whose argmax places peak
  blobs at known integer coordinates, plus a `List[List[Tuple[int, int]]]` of
  those coordinates per panel. Class 0 is background, class 1 is peak
  (matches `cxi_pipeline_ray.core.peak_finding`). When `peak_locations` is
  `None`, peaks are sampled via `np.random.default_rng(seed)` with a
  rejection-sampling loop that enforces minimum spacing so the blobs do not
  merge under scipy's 8-connectivity labeling.

- `make_synthetic_pipeline_output(B, C, H_orig, W_orig, H_preprocessed, W_preprocessed, peaks_per_panel=3, peak_locations=None, peak_blob_size=3, photon_wavelength=1.3, timestamp=0, seed=0, draw_peaks_in_image=True) -> (FakePipelineOutput, ground_truth)`
  — Tier 2 extension that also builds a synthetic detector-image tensor
  (`ray.put`-ed so the writer can reconstruct from it) and physics metadata
  (`photon_wavelength`, `timestamp`). The returned `ground_truth` is clipped
  to the original (pre-padding) detector shape, mirroring the bottom-right
  padding assumption in `coordinator.process_batch`.

- `FakePipelineOutput(logits, preprocessing_metadata=None, original_image_ref=None, metadata=None)`
  — duck-typed stand-in for a real `PipelineOutput`. Exposes the attributes
  the coordinator reads (`preprocessing_metadata`, `original_image_ref`,
  `metadata`) and a `get_torch_tensor(device='cpu')` method that imports
  torch lazily — falling back to a numpy-shim object if torch is not
  installed in the dev venv.

Tier 2 tests also use a small `FakePreprocessingMetadata` dataclass
(`original_shape`, `preprocessed_shape`) instantiated internally by
`make_synthetic_pipeline_output`.

## Performance Tuning

### Symptoms and Solutions

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| Low CPU utilization (<50%) | Underutilized CPUs | Increase `num_cpu_workers` |
| High task scheduling overhead | Too many tiny tasks | Decrease `num_cpu_workers` |
| Q2 queue growing | Writer too slow | Increase `num_cpu_workers` |
| Memory usage growing | Too many pending tasks | Decrease `max_pending_tasks` |
| Too many small CXI files | Buffer flushing too often | Increase `buffer_size` |

### Monitoring

```bash
# Ray dashboard
ray dashboard

# Monitor writer logs
cxi-writer --config my_config.yaml --log-level DEBUG

# Monitor CXI output
watch -n 5 'ls -lh /output/dir/*.cxi'
```

## Development

### Running Tests

See the [Testing](#testing) section above for the three-tier test suite
(pytest fixtures, offline writer end-to-end, and Q2 injection benchmark).

### Code Style

```bash
# Format code
black cxi_pipeline_ray/

# Lint
ruff check cxi_pipeline_ray/
```

## Troubleshooting

### Common Issues

**Ray namespace mismatch**
```
Error: Cannot find queue 'peaknet_q2' in namespace 'peaknet-pipeline'
```
Solution: Check `ray.namespace` matches in both pipeline and writer configs

**Queue shards mismatch**
```
Error: Queue 'peaknet_q2' has 4 shards, expected 8
```
Solution: Check `queue.num_shards` matches pipeline's `queue_num_shards`

**Geometry file not found**
```
FileNotFoundError: /path/to/detector.geom
```
Solution: Verify `geometry.geom_file` path or set to `null` to skip conversion

**OOM (Out of Memory)**
```
Ray ObjectStoreFullError: ...
```
Solution: Decrease `max_pending_tasks` or increase Ray object store size

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please contact:
- Cong Wang <cwang31@slac.stanford.edu>
