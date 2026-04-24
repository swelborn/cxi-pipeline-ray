# Post-Fix Benchmark — Light Config, K=4 (speedup check)

**Task:** 26 — verify Task 25's Axis-1 fix delivers parallelism when
`num_cpu_workers > 1`.
**Status:** POST-FIX. `num_cpu_workers: 4` activates the `@ray.remote`
fan-out in `coordinator.process_batch` at line 283.
**Companion reports:** see `2026-04-22-after-light-k1.md` and summary.md.

---

## Command

```bash
docs/benchmarks/run-bench-post.sh after-light-k4 200 20 256 256 3 \
  docs/benchmarks/bench-writer-k4.yaml
```

Writer config delta vs K=1: `processing.num_cpu_workers: 4`. Since
`B*C = 1*4 = 4` panels per batch, K=4 matches the effective ceiling
(each panel becomes one `_find_peaks_task.remote`, so K >= num_panels
yields the same fan-out). K values above 4 would add cluster CPU
reservation but not extra parallelism for this batch shape; see the
decision doc (docs/design/bottleneck-fix-decision.md §"Effective
peak-finding parallelism") for the ceiling analysis.

## Code under test

Same as K=1 run: branch `feat/fix-bottleneck`, SHA `2f0edb7`.

## Machine

Same as K=1 run: sdfada002, JOBID 25663264, 96 logical CPU host, 288 GB
allocated.

## Bench report numbers

```
num_batches_pushed           : 200
batches_per_second_target    : 20.000
batches_per_second_achieved  : 20.000
producer_wall_seconds        : 10.000
drain_wall_seconds           : 0.002
end_to_end_wall_seconds      : 10.002
inter_push_p50_seconds       : 0.050000
inter_push_p95_seconds       : 0.050012
inter_push_p99_seconds       : 0.077638
queue_depth_max              : 5
cxi_files_written            : 20
cxi_events_written           : 200
events_per_batch_expected    : 1
events_total_expected        : 200
```

### Effective sustained writer throughput

```
200 batches / 10.002 s  ≈  20.0 batches/sec
```

### Comparison to K=1

| Metric | K=1 | K=4 | Δ |
|---|---|---|---|
| drain_wall_seconds | 0.002 | 0.002 | 0% |
| queue_depth_max | 8 | 5 | -3 |
| p99 inter_push | 0.0948 | 0.0776 | -18% |

At 256x256 the writer's single-batch cost is so small (~0.3 ms/panel, ~1 ms
total with overhead) that Axis-1 fan-out barely moves the needle on the
drain time — both runs are producer-limited. The small queue-depth
improvement at K=4 is consistent with Ray task scheduling absorbing the
occasional slow batch in parallel. Speedup at this config is therefore
**not measurable** beyond noise; the realistic config is the rigorous test
for speedup.

## Verdict

**Light config: AMBIGUOUS / producer-limited.** Both K=1 and K=4 keep up
with the 20 bps producer; drain is < 3 ms in both cases. This confirms
Axis-1 fan-out does not regress at small batch size but is not the
throughput test — see `2026-04-22-after-realistic-k4.md`.

## Artifacts

- Log: `/sdf/scratch/users/c/cwang31/bench-out/after-light-k4.log`
- CXI output: `/sdf/scratch/users/c/cwang31/bench-out/after-light-k4/*.cxi`
  (20 files)
- Writer config: `docs/benchmarks/bench-writer-k4.yaml`
