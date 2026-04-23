# Baseline Benchmark — Realistic Config (1696x1696)

**Task:** 22 — quantify where the single-threaded bottleneck in the CXI writer lives.
**Status:** Baseline, UNMODIFIED code. No Axis-1/2 changes applied yet.
**Companion report:** `2026-04-22-baseline-light.md` (256x256, writer keeps up).

---

## Command

```bash
docs/benchmarks/run-bench-profiled-v2.sh baseline-realistic 500 50 1696 1696 30
```

Which expands to:

```bash
python tests/bench_q2_inject.py \
  --num-batches 500 \
  --batches-per-second 50 \
  --peaks-per-panel 30 \
  --B 1 --C 4 \
  --H-orig 1696 --W-orig 1696 \
  --H-preprocessed 1696 --W-preprocessed 1696 \
  --output-dir /sdf/scratch/users/c/cwang31/bench-out/baseline-realistic \
  --writer-config docs/benchmarks/bench-writer.yaml \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 600 \
  --writer-startup-seconds 3 \
  --log-level INFO
```

Image shape deviates from the canonical demo (1667×1668 original
padded to 1696×1696); the benchmark uses H_orig == H_preprocessed so the
bottom-right clip is a no-op. This does not affect the writer bottleneck
— peak-finding still operates on the full 1696×1696 logit map per panel.

py-spy was attached to the writer subprocess for the middle ~25s of the run
(`rate=200`, `--subprocesses`, speedscope output).

## Code under test

- Repo: `cxi-pipeline-ray`
- Branch: `feat/fix-bottleneck`
- Git SHA: `aaa7cf2` (`aaa7cf2862cbaffc0a87e5a62e81cf6a24f08aee`) — tip of
  `feat/fix-bottleneck` at run time, identical writer code to `main`.

## Machine

- Host: `sdfada002` (SLAC SDF, `ada` partition)
- CPU: AMD EPYC 7713P 64-Core (96 logical CPUs; Ray reported `CPU: 96.0`)
- GPU: 4× NVIDIA L40S, 46 GB — **not used** by this benchmark
- RAM: 703 GB (288 GB allocated)
- OS: RHEL 8.6, kernel 4.18.0-372.32.1
- Allocation: `salloc --no-shell --account=lcls:prjdat21 --partition=ada
  --nodes=1 --gpus=4 --time=4:00:00` (JOBID 25663264)
- Writer runs natively inside the alloc (no container).

## Bench report numbers

```
num_batches_pushed           : 500
batches_per_second_target    : 50.000
batches_per_second_achieved  : 50.000   ← producer rate (push into Q2)
producer_wall_seconds        : 10.000
drain_wall_seconds           : 394.448  ← writer-bound time
end_to_end_wall_seconds      : 404.448
inter_push_p50_seconds       : 0.019999
inter_push_p95_seconds       : 0.020021
inter_push_p99_seconds       : 0.084817
queue_depth_max              : 490      ← producer fills Q2 immediately
cxi_files_written            : 50
cxi_events_written           : 500
events_per_batch_expected    : 1
events_total_expected        : 500
```

### Effective sustained writer throughput

```
500 batches / 404.448 s  ≈  1.24 batches/sec
```

That is **~40× below the 50 batches/sec producer target.** The producer
hits its target trivially (it just pushes synthetic tensors into a Ray
queue); the writer is the bottleneck.

### Per-batch processing (writer side, coordinator log timestamps)

The coordinator log stamps only at 1 s granularity, so individual p95/p99
values read directly from timestamps are quantized. Values are consistent
with the end-to-end number, however:

- n = 499 inter-batch deltas
- **mean = 810 ms per batch** (= 1 / 1.236 bps, confirms drain math above)
- Per-batch times drift from ~700 ms early to ~1.0–2.0 s late in the run
  as queue depth grows — consistent with Ray object-store memory pressure
  (queue_depth_max = 490 batches × ~46 MB logits per batch ≈ 22 GB in
  flight) and GC/allocator overhead on the writer.

## py-spy dominant-stage numbers

File: `baseline-realistic.speedscope.json`, total weight **23.285 s** of
sampled writer CPU time (2 profiles — main + single worker thread of the
writer process; 4 657 samples @ 200 Hz).

**Top by SELF time (where the CPU *was*):**

| % self | Function | Meaning |
|---|---|---|
| **75.63%** | `find_peaks_numpy` (leaf at peak_finding.py:41) | center-of-mass / per-component loop inside peak finding |
| 15.25% | `numpy._wrapfunc` | called exclusively from `np.argmax(logits, axis=0)` at peak_finding.py:28 |
|  3.87% | `find_peaks_numpy` (line 40) | labelled-component iteration scaffolding |
|  3.56% | `scipy.ndimage.label` | 8-connectivity component labeling |
|  0.47% | `find_peaks_numpy` (line 28) | the argmax call itself |
|  0.39% | `ray.actor._actor_method_call` | writer-actor submission |
|  0.21% | `ray._private.worker.get_objects` | Q2 dequeue |

**Top by TOTAL time (paths through which time flowed):**

- `process_batch` (coordinator.py) = **99.76%**
- `find_peaks_numpy` = **99.12%**
- `np.argmax` / `_wrapfunc` path = 15.25%
- `scipy.ndimage.label` path = 3.61%
- `ray.actor._remote` + `_actor_method_call` (submit to writer actor) = 0.41%
- `ray.get_objects` (Q2 dequeue) = 0.21%

CXI file flush / h5py / HDF5 write paths do not appear in the top-25 TOTAL
table at all — they are below ~0.5% noise.

**Per-batch decomposition (using total-time shares against 810 ms/batch mean):**

| Stage | % of batch time | Absolute per-batch |
|---|---|---|
| `find_peaks_numpy` total | 99.12% | ≈ 803 ms |
|   ↳ center-of-mass loop (leaf 75.63%) | | ≈ 613 ms |
|   ↳ `np.argmax` over 2×1696×1696 logits | | ≈ 124 ms |
|   ↳ `scipy.ndimage.label` 8-conn | | ≈ 29 ms |
|   ↳ other peak_finding.py scaffolding | | ≈ 37 ms |
| Ray Q2 dequeue + actor submit | 0.62% | ≈ 5 ms |
| CXI flush / h5py / HDF5 write | < 0.5% | < 4 ms |

## Verdict

**PEAK FINDING DOMINATES.**

- `find_peaks_numpy` accounts for **99.12%** of observed writer stack time
  and **~803 ms of the 810 ms per-batch mean** — a 200× margin over any
  other stage.
- CXI / HDF5 write cost is invisible at < 0.5% of total stack time and
  would correspond to < 4 ms per batch.
- Within `find_peaks_numpy`, the dominant contributor is the
  post-labeling loop that computes center-of-mass per component (leaf
  at peak_finding.py:41, 75.6% self), followed by the single
  `np.argmax(logits, axis=0)` call over the full 2×1696×1696 map.

This satisfies the Task 22 acceptance gate ("at least one report shows a
clear single-stage-dominant verdict backed by numbers — stage X accounts
for >60% of total per-batch time"): peak finding is 99% by every measure.

### Implications for the axis decision (Task 24 input)

Per the decision matrix in Task 24:

- If sharded CXI output is OK (Task 23 verdict SHARDED OK) → **Axis 1**
  (smaller diff than Combined; predicted speedup ≈ min(K, num_panels) =
  min(K, 4) for B=1, C=4 panels per batch).
- If sharded CXI output is NOT OK (verdict SINGLE-FILE REQUIRED) → **Axis 1**
  anyway, since peak finding is 99% of the work — Axis 2 would buy
  essentially zero, and sharding files for a non-sharding-friendly
  downstream is unnecessary pain.
- **Axis 2 is not a useful fix here.** CXI write cost is < 0.5% of total
  time; sharding across N writer actors would save < 4 ms per batch
  while multiplying output file count by N.
- **Combined (A + B) buys almost nothing over A alone** for the same
  reason.

### Residual risks to track in Task 25

1. **Ray task serialization overhead for Axis 1.** Each peak-finding task
   would serialize a (2, 1696, 1696) float32 ndarray — 22 MB per panel —
   through Ray's object store. At B=1, C=4, that is 4 tasks × 22 MB = 88
   MB per batch. If the fan-out/fan-in overhead exceeds the 803 ms
   currently spent serially, the fix regresses.
2. **Per-panel workload granularity.** B=1, C=4 means only 4 panels per
   batch. If num_cpu_workers > 4, excess workers sit idle; effective
   parallelism is capped at C. Document this in `cxi_writer.yaml` so
   operators don't set num_cpu_workers=16 expecting linear gain.
3. **Scipy `label` is already threaded via OpenMP?** Check whether
   `scipy.ndimage.label` with `structure=np.ones((3,3))` respects
   `OMP_NUM_THREADS`. If yes, Axis 1 may not help as much as the
   top-level split predicts because each task already pulls CPU cycles
   from the same pool.

## Artifacts

- Log: `/sdf/scratch/users/c/cwang31/bench-out/baseline-realistic.log`
- py-spy: `/sdf/scratch/users/c/cwang31/bench-out/baseline-realistic.speedscope.json`
- CXI output: `/sdf/scratch/users/c/cwang31/bench-out/baseline-realistic/*.cxi`
  (50 files, ~439 MB each — 500 events, ~22 GB total; dominated by the
  1696×1696 float32 detector image, not peak data)
- Parser: `docs/benchmarks/parse-pyspy.py`
