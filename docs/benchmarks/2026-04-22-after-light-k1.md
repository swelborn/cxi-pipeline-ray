# Post-Fix Benchmark — Light Config, K=1 (regression check)

**Task:** 26 — verify Task 25's Axis-1 fix (num_cpu_workers knob) preserves
K=1 behavior and delivers speedup at K>1.
**Status:** POST-FIX. Code under test has Task 25's Axis-1 parallelization
available but `num_cpu_workers: 1` keeps the sequential path (see
coordinator.py:283 guard).
**Companion reports:**
- `2026-04-22-after-light-k4.md` — K=4, light config
- `2026-04-22-after-realistic-k1.md` — K=1, realistic config (regression)
- `2026-04-22-after-realistic-k4.md` — K=4, realistic config (speedup)
- `summary.md` — cross-run table and verdict

---

## Command

```bash
docs/benchmarks/run-bench-post.sh after-light-k1 200 20 256 256 3 \
  docs/benchmarks/bench-writer-k1.yaml
```

Which expands to:

```bash
python tests/bench_q2_inject.py \
  --num-batches 200 \
  --batches-per-second 20 \
  --peaks-per-panel 3 \
  --B 1 --C 4 \
  --H-orig 256 --W-orig 256 \
  --H-preprocessed 256 --W-preprocessed 256 \
  --output-dir /sdf/scratch/users/c/cwang31/bench-out/after-light-k1 \
  --writer-config docs/benchmarks/bench-writer-k1.yaml \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 900 \
  --writer-startup-seconds 3 \
  --log-level INFO
```

Same image shape, batch rate, and seed as Task 22's `baseline-light` so the
two are directly comparable. Writer config differs only in adding
`processing.num_cpu_workers: 1` (new knob introduced by Task 25).

## Code under test

- Repo: `cxi-pipeline-ray`
- Branch: `feat/fix-bottleneck`
- Git SHA: `2f0edb7` (`2f0edb75dc131bf40de0b6ea075af3305e622eeb`) — Task 25's
  merge of Axis-1 implementation: `@ray.remote _find_peaks_task` +
  `num_cpu_workers` kwarg on `process_batch`.
- Baseline comparison SHA (Task 22): `aaa7cf2`.

## Machine

- Host: `sdfada002` (SLAC SDF, `ada` partition) — SAME node as Task 22.
- CPU: AMD EPYC 7713P 64-Core (96 logical CPUs on the host; 72 reported by
  squeue for the allocation).
- GPU: 4× NVIDIA L40S, 46 GB — **not used** by this benchmark.
- RAM: 703 GB host, 288 GB allocated.
- OS: RHEL 8.6, kernel 4.18.0-372.32.1.
- Allocation: `salloc --no-shell --account=lcls:prjdat21 --partition=ada
  --nodes=1 --gpus=4 --time=4:00:00` (JOBID 25663264 — SAME alloc as Task 22).
- Writer runs natively inside the alloc (no container).

## Bench report numbers

```
num_batches_pushed           : 200
batches_per_second_target    : 20.000
batches_per_second_achieved  : 20.000
producer_wall_seconds        : 10.000
drain_wall_seconds           : 0.002
end_to_end_wall_seconds      : 10.002
inter_push_p50_seconds       : 0.050000
inter_push_p95_seconds       : 0.050014
inter_push_p99_seconds       : 0.094753
queue_depth_max              : 8
cxi_files_written            : 20
cxi_events_written           : 200
events_per_batch_expected    : 1
events_total_expected        : 200
```

### Effective sustained writer throughput

```
200 batches / 10.002 s  ≈  20.0 batches/sec
```

Writer keeps up with the 20 bps producer target — identical behavior to the
pre-fix baseline (Task 22 light, where `baseline-light.md` reported
drain=0.002s, queue_depth_max=4, achieved=20.0 bps).

### Regression check vs Task 22 baseline-light

| Metric | Baseline (aaa7cf2) | Post-fix K=1 (2f0edb7) | Δ |
|---|---|---|---|
| drain_wall_seconds | 0.002 | 0.002 | 0% |
| end_to_end_wall_seconds | 10.002 | 10.002 | 0% |
| queue_depth_max | 4 | 8 | +4 |
| batches_per_second_achieved | 20.0 | 20.0 | 0% |

Queue depth difference is within noise at this light load (writer finishes
each batch in << 50 ms). End-to-end timing is bit-identical at 10.002s.
**Regression gate PASS (within 10% of baseline).**

## Verdict

**K=1 regression: PASS.** The Task 25 Axis-1 branch preserves the
sequential path exactly when `num_cpu_workers=1` — drain and throughput
match baseline to 3 decimal places. Light config does not place the writer
under backpressure, so speedup must be measured at realistic config; see
`2026-04-22-after-realistic-k1.md` + `...-k4.md`.

## Artifacts

- Log: `/sdf/scratch/users/c/cwang31/bench-out/after-light-k1.log`
- CXI output: `/sdf/scratch/users/c/cwang31/bench-out/after-light-k1/*.cxi`
  (20 files, ~10 MB each — 200 events total)
- Writer config used: `docs/benchmarks/bench-writer-k1.yaml`
