# Benchmark Summary — CXI Writer Single-Threaded Bottleneck Fix

**Task 26** verification of Task 25's Axis-1 parallelization
(`processing.num_cpu_workers` knob).

## Headline

Task 25's `@ray.remote _find_peaks_task` fan-out in
`cxi_pipeline_ray.core.coordinator.process_batch` delivers a
**2.86-2.92x throughput speedup** on production image shapes
(1696x1696 × B=1 × C=4 panels) with no regression at K=1.

## Results table

| Config | num_cpu_workers (K) | Batches | Achieved rate (bps) | p95 inter-push (ms) | Speedup vs baseline (aaa7cf2) | Speedup vs K=1 (2f0edb7) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Light (256^2) | 1 | 200 | 20.00 | 50.01 | 0% (baseline also kept up) | — | regression-PASS |
| Light (256^2) | 4 | 200 | 20.00 | 50.01 | 0% (producer-limited) | ~0% (noise) | no-backpressure, not a speedup test |
| Realistic (1696^2) | 1 (500-batch) | 500 | 1.254 | 20.01 | +1.5% | — | regression-PASS (within 10%) |
| Realistic (1696^2) | 1 (200-batch) | 200 | 1.260 | 24.16 | +1.9% | — | regression-PASS |
| Realistic (1696^2) | 4 | 200 | **3.605** | 31.72 | **2.92x** vs Task 22 baseline | **2.86x** vs K=1 2f0edb7 | **SPEEDUP ACHIEVED** |

Baseline comparison target: `2026-04-22-baseline-realistic.md`
(SHA aaa7cf2, 500-batch at 50 bps) reported 1.236 bps.

## Regression gate (K=1 must stay within 10% of baseline)

| Metric | Baseline aaa7cf2 | Post-fix K=1 (2f0edb7, 500 batches) | delta | PASS? |
|---|---:|---:|---:|:---:|
| drain_wall_seconds | 394.448 | 388.689 | -1.46% | yes |
| end_to_end_wall_seconds | 404.448 | 398.690 | -1.42% | yes |
| queue_depth_max | 490 | 488 | -0.4% | yes |
| achieved batches/sec | 1.236 | 1.254 | +1.5% | yes |

**PASS** — K=1 post-fix is within 2% of baseline on every metric.
This confirms the guard at `coordinator.py:283`
(`if num_cpu_workers > 1 and num_panels > 1`) correctly routes K=1
through the sequential path with no Ray overhead.

## Speedup vs Task 24's back-of-envelope prediction

Task 24's decision doc (`docs/design/bottleneck-fix-decision.md`) predicted
`min(K, num_panels)` for peak-finding-bound runs.

- Predicted: min(4, 4) = 4.0x
- Observed: 2.92x (vs baseline) / 2.86x (vs K=1 post-fix)
- Shortfall: ~27%

Characterized as Ray task dispatch overhead. Each panel's
1696x1696x2-class logit tensor serializes to ~23 MB over the Ray
object store per `.remote()` call. With ~202 ms of theoretical
compute per panel (810 ms / 4) and the observed per-batch drain
of ~257 ms, the ~55 ms gap is consistent with object-store
serialization of 4 panels. See
`2026-04-22-after-realistic-k4.md` for the detailed breakdown.

## K selection note for operators

For the current demo's batch shape (B=1, C=4 -> 4 panels per batch),
the effective parallelism ceiling is **4**. Setting
`num_cpu_workers` higher than 4 reserves extra CPUs without adding
throughput. If future work raises C or B, K can be tuned upward
accordingly.

## Cluster environment

All runs executed inside SLURM jobid 25663264 on sdfada002
(SLAC SDF `ada` partition), 96 logical CPUs, 703 GB RAM, 4x L40S
(GPU unused). Allocation held across all 5 runs — no inter-run
machine drift.

## Reports

- `2026-04-22-after-light-k1.md`
- `2026-04-22-after-light-k4.md`
- `2026-04-22-after-realistic-k1.md` (covers both 500-batch and 200-batch K=1 runs)
- `2026-04-22-after-realistic-k4.md`

Baseline references:
- `2026-04-22-baseline-light.md` (Task 22)
- `2026-04-22-baseline-realistic.md` (Task 22)

Decision trail:
- `../design/bottleneck-fix-decision.md` (Task 24, CHOSEN AXIS: Axis 1)
- `../design/downstream-cxi-compatibility.md` (Task 23, SHARDED OK)
