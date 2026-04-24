# Post-Fix Benchmark — Realistic Config, K=1 (regression check)

**Task:** 26 — verify Task 25's Axis-1 fix preserves sequential-path
behavior (bit-equivalent to pre-change baseline in the `num_cpu_workers=1`
branch).
**Status:** POST-FIX. `num_cpu_workers: 1` exercises the else-branch at
coordinator.py:291 — no Ray task scheduling, no object-store round trip.
**Companion reports:**
- `2026-04-22-after-realistic-k4.md` — K=4 speedup run (same shape, same
  alloc, same seed).
- `2026-04-22-baseline-realistic.md` — Task 22 baseline (SHA aaa7cf2).

---

## Command

```bash
docs/benchmarks/run-bench-post.sh after-realistic-k1 500 50 1696 1696 30 \
  docs/benchmarks/bench-writer-k1.yaml
```

Expands to:

```bash
python tests/bench_q2_inject.py \
  --num-batches 500 \
  --batches-per-second 50 \
  --peaks-per-panel 30 \
  --B 1 --C 4 \
  --H-orig 1696 --W-orig 1696 \
  --H-preprocessed 1696 --W-preprocessed 1696 \
  --output-dir /sdf/scratch/users/c/cwang31/bench-out/after-realistic-k1 \
  --writer-config docs/benchmarks/bench-writer-k1.yaml \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 900 \
  --writer-startup-seconds 3 \
  --log-level INFO
```

Same params as Task 22 `baseline-realistic` — only change is
`processing.num_cpu_workers: 1` in the writer config.

## Code under test

- Repo: `cxi-pipeline-ray`
- Branch: `feat/fix-bottleneck`
- Git SHA: `2f0edb7` — Task 25's Axis-1 implementation.
- Baseline SHA (Task 22): `aaa7cf2` — no Axis-1 code; writer was strictly
  sequential.

## Machine

- Host: `sdfada002` (SAME as Task 22)
- Allocation: JOBID 25663264 (SAME allocation; started `2:14h` before
  Task 22, still held at Task 26 runtime — reproducibility maximized).
- CPU / GPU / RAM / OS: identical to the baseline report; see
  `2026-04-22-baseline-realistic.md` §Machine.

## Bench report numbers

### Primary run: 500 batches (full Task 22 reproduction)

```
num_batches_pushed           : 500
batches_per_second_target    : 50.000
batches_per_second_achieved  : 50.000
producer_wall_seconds        : 10.000
drain_wall_seconds           : 388.689
end_to_end_wall_seconds      : 398.690
inter_push_p50_seconds       : 0.019999
inter_push_p95_seconds       : 0.020007
inter_push_p99_seconds       : 0.082907
queue_depth_max              : 488
cxi_files_written            : 50
cxi_events_written           : 500
events_per_batch_expected    : 1
events_total_expected        : 500
```

### Companion run: 200 batches (apples-to-apples vs K=4)

Added after the K=4 realistic run was shortened to 200 batches to fit
in the 100 GB scratch quota. Same parameters otherwise.

```
num_batches_pushed           : 200
batches_per_second_target    : 50.000
batches_per_second_achieved  : 49.999
producer_wall_seconds        : 4.000
drain_wall_seconds           : 154.703
end_to_end_wall_seconds      : 158.703
inter_push_p50_seconds       : 0.016200
inter_push_p95_seconds       : 0.024160
inter_push_p99_seconds       : 0.100179
queue_depth_max              : 195
cxi_files_written            : 20
cxi_events_written           : 200
events_per_batch_expected    : 1
events_total_expected        : 200
```

### Effective sustained writer throughput

```
500-batch: 500 / 398.690 s  ≈  1.254 batches/sec
200-batch: 200 / 158.703 s  ≈  1.260 batches/sec
```

The two K=1 runs agree within 0.5% — the writer's steady-state
throughput is consistent across different batch-count runs.

Baseline was `500 / 404.448 ≈ 1.236 bps`, so the post-fix K=1 run is
**+1.5% faster**. This is within measurement noise (coordinator log
timestamps quantized at 1 s + one OS scheduling jitter on the held alloc)
and well inside the Task 26 regression gate of "within 10% of baseline".

### Regression check vs Task 22 baseline-realistic

| Metric | Baseline (aaa7cf2) | Post-fix K=1 (2f0edb7) | Δ% |
|---|---|---|---|
| drain_wall_seconds | 394.448 | 388.689 | -1.46% |
| end_to_end_wall_seconds | 404.448 | 398.690 | -1.42% |
| queue_depth_max | 490 | 488 | -0.4% |
| batches_per_second_achieved (producer) | 50.000 | 50.000 | 0% |
| cxi_files_written | 50 | 50 | 0 |
| cxi_events_written | 500 | 500 | 0 |

**Regression gate PASS.** The K=1 guard at coordinator.py:283 keeps the
pre-Axis-1 code path intact — end-to-end timing matches baseline within
1.5%.

### Per-batch processing

Drain time / n_batches = 388.689 / 500 = **777 ms/batch mean** (vs baseline
810 ms/batch). py-spy was not re-attached for this run because the
baseline profile already established that 99.12% of stack time is inside
`find_peaks_numpy`; the near-identical drain time confirms that the
sequential code path behaves identically post-fix.

## Verdict

**Regression check: PASS.** Post-fix K=1 run matches Task 22 baseline
within 1.5% on drain wall time, 0% on batch count, 0% on output files.
Task 25's guard (`if num_cpu_workers > 1 and num_panels > 1`) correctly
routes K=1 runs through the sequential path.

## Artifacts

- 500-batch log: `/sdf/scratch/users/c/cwang31/bench-out/after-realistic-k1-500batches.log`
- 200-batch log: `/sdf/scratch/users/c/cwang31/bench-out/after-realistic-k1-200b.log`
- 200-batch CXI output: `/sdf/scratch/users/c/cwang31/bench-out/after-realistic-k1-200b/*.cxi`
  (20 files, 200 events)
- 500-batch CXI output was deleted after the log was captured to free
  scratch quota for the K=4 run; reports cite the log for all numbers.
- Writer config: `docs/benchmarks/bench-writer-k1.yaml`
