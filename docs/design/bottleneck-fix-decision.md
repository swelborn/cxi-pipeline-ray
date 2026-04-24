# Bottleneck Fix — Axis Decision

**Task:** 24 — pick the parallelization strategy for the CXI writer fix so
that Task 25 has zero judgment calls left.

**Inputs:**
- `docs/benchmarks/2026-04-22-baseline-light.md` (Task 22, 256×256)
- `docs/benchmarks/2026-04-22-baseline-realistic.md` (Task 22, 1696×1696)
- `docs/design/downstream-cxi-compatibility.md` (Task 23)

---

## CHOSEN AXIS

**Axis 1.**

One writer actor. Peak finding inside `cxi_pipeline_ray/core/coordinator.py::process_batch`
is fanned out across `num_cpu_workers` Ray tasks, one per panel
(`logits.shape[0]` = `B * C`). The file writer, CXI/HDF5 layout, and
downstream file-shape contract are all unchanged.

---

## Rationale

### 1. The bottleneck is peak finding, not CXI write

From `docs/benchmarks/2026-04-22-baseline-realistic.md` (1696×1696,
production shape, JOBID 25663264 on `sdfada002`):

| Stage | Total-time share | Absolute per-batch |
|---|---|---|
| `find_peaks_numpy` (total) | **99.12%** | ≈ 803 ms of 810 ms mean |
| Ray Q2 dequeue + actor submit | 0.62% | ≈ 5 ms |
| CXI flush / h5py / HDF5 write | **< 0.5%** | < 4 ms |

(Source: `2026-04-22-baseline-realistic.md` §"Per-batch decomposition",
lines 132–142. py-spy profile `baseline-realistic.speedscope.json`,
23.285 s of writer CPU time, 4 657 samples @ 200 Hz.)

`process_batch` total-time share is 99.76% — everything outside
peak-finding-driven `process_batch` is noise. Effective sustained writer
throughput is ≈ 1.24 batches/sec against a 50 bps producer (40× under
target, line 82 of the report). The 256×256 light run
(`2026-04-22-baseline-light.md` lines 109–123) corroborates: even when
the writer keeps up, what CPU time it does spend is 60.5% in
`find_peaks_numpy` vs. < 0.5% in CXI write. The ratio only sharpens at
production shape.

This triggers branch (i) of the Task-24 decision matrix
("peak finding dominates AND sharded output is OK → Axis 1 or Combined,
prefer Axis 1 for smaller diff") *and* — importantly — also picks
Axis 1 under branch (iv) ("sharded NOT OK → Axis 1 only"), because the
cost of Axis 2 buys effectively zero (< 4 ms per batch) whether
sharding is acceptable or not. The sharding verdict does not change
the answer here.

### 2. Sharding verdict (Task 23): SHARDED OK (provisional)

`docs/design/downstream-cxi-compatibility.md` lines 14–27:

> Every CXI-consuming artifact in `/sdf/data/lcls/ds/prj/prjcwang31/results/proj-stream-to-ml`
> either operates on a single file supplied by the caller, enumerates
> files from a directory listing, or is already written against the
> existing chunked output (filenames of the form
> `peaknet_cxi_<timestamp>_chunk0000.cxi`, `_chunk0001.cxi`, …).
> …An Axis-2 change would multiply `N` by the number of writer actors
> — it does not introduce file-plurality where there was none.

Two stakeholder rows (bottleneck reporter, downstream indexing owner)
are still PENDING (`downstream-cxi-compatibility.md` lines 114–118).
This is why Task 23's verdict is marked *provisional* — but it does not
affect this decision, because peak finding is the 99% hitter and Axis 1
is the right choice under either verdict.

### 3. Why not Axis 2 or Combined

- **Axis 2 alone** would parallelize across N writer actors. With CXI
  write cost at < 0.5% of total stack time (< 4 ms / 810 ms batch), the
  theoretical speedup is ≤ 1.005× regardless of N. Meanwhile the
  diff is larger (cli.py fan-out + coordinator round-robin + test
  matrix × N actors). The cost/benefit is inverted.
- **Combined (A + B)** buys almost nothing over A alone for the same
  reason: the stage that B parallelizes is already invisible in the
  profile. Combined is only a win when *both* stages are substantial
  fractions of total time; here they are not (99% vs < 0.5%).

### 4. Quantitative speedup estimate (back-of-envelope)

For Axis 1 the ceiling is `min(K, num_panels_per_batch)`, where
`num_panels_per_batch = B * C`. In the realistic demo config
(`demo/configs/cxi_writer.yaml` and `peaknet.yaml`: `B = 1`, `C = 4`
for an ePix100 4-panel detector), this is `min(K, 4)`.

If Ray task overhead is negligible:

| num_cpu_workers (K) | Effective peak-finding parallelism | Predicted per-batch time |
|---|---|---|
| 1 (baseline) | 1 | 803 ms (measured) |
| 2 | 2 | ≈ 402 ms |
| 4 | 4 | ≈ 201 ms |
| 8, 16, 32 | 4 (capped by C) | ≈ 201 ms |

That is a **~4× ceiling** for the current demo shape. Driving
throughput above that requires growing `B` or `C`, changing the image
size, or switching algorithm — all out of scope for Task 25.

The ceiling is honest about what this fix can and cannot do:
- It is enough to clear the 50 bps producer target (`50 * 201 ms = 10.05 s`
  mean write CPU per 10 s of producer → still saturates, but only
  ~barely). If the realistic target is "keep up with ePix100 at
  120 Hz" the ceiling is insufficient and a further algorithmic fix
  (vectorized peak finding, GPU peak finding, or larger batching) will
  be needed after Axis 1 lands.
- But it is a monotonic improvement: `num_cpu_workers=1` ≡ current
  behavior (regression gate), and any `K > 1` is strictly faster in
  the best case.

### 5. Consistency with Task 22's own recommendation

`baseline-realistic.md` lines 162–177 already names Axis 1 as the
implied choice from the profile ("CXI / HDF5 write cost is invisible
at < 0.5% of total stack time … Axis 2 is not a useful fix here …
Combined (A + B) buys almost nothing over A alone"). This decision
doc inherits those conclusions and binds them to the Task 23 verdict.

---

## Risks

1. **Ray task serialization overhead may eat the gain at small B × C.**
   Each `_find_peaks_task.remote(panel_logits, …)` serializes a
   `(2, H, W)` float32 ndarray — for 1696×1696, that is 22 MB per panel
   through Ray's object store. At B=1, C=4, the batch-wide serialization
   cost is 4 × 22 MB = 88 MB per batch. If the object-store round-trip
   per task exceeds ~150 ms (roughly the gain we want from parallel
   split at K=4), the fix regresses on small batches.
   **Mitigation:** benchmark num_cpu_workers ∈ {1, 2, 4, 8, 16} against
   Task 22's realistic config in Task 26 and pick the K that actually
   wins. If K=1 is within 10% of the existing baseline but no K > 1
   wins, revert and consider an in-process `ThreadPoolExecutor` fallback
   instead of Ray tasks (note this as future work, not in scope for
   Task 25).
   (Source for 22 MB figure: `baseline-realistic.md` lines 181–185.)

2. **Parallelism ceiling = num_panels_per_batch (= 4 for current demo).**
   Operators who set `num_cpu_workers=16` expecting linear scaling will
   get at most 4×. This must be documented in
   `examples/configs/cxi_writer_default.yaml` and the README knob
   description — the existing README at README.md:98 calls it "Parallel
   Ray tasks (default: 16)" which is misleading in both directions
   (phantom at rest, capped in practice).
   **Mitigation:** Task 25 updates README.md:98 and the yaml comments
   at examples/configs/cxi_writer_default.yaml:44, 178, 185, 201 to
   state the true ceiling: `min(num_cpu_workers, B * C)`.

3. **Hidden OMP parallelism inside the dominant stages.** `np.argmax`
   over 2×1696×1696 logits (15.25% of time per
   `baseline-realistic.md` line 113) and `scipy.ndimage.label`
   (3.56% per line 115) may already multi-thread via OpenMP. If so,
   fanning out K Ray tasks that each internally pull
   `OMP_NUM_THREADS` CPUs will starve the thread pool and the effective
   speedup is < K. This is the same concern Task 22 flagged
   (`baseline-realistic.md` lines 190–194).
   **Mitigation:** Task 25 adds `os.environ['OMP_NUM_THREADS'] = '1'`
   at the top of `_find_peaks_task` (or passes `OMP_NUM_THREADS=1`
   through `ray.remote(runtime_env=...)` if that is cleaner). Task 26
   benchmarks verify the chosen knob actually improves throughput.

4. **Ray cluster CPU budget.** The existing `ray.init()` in
   `cxi_pipeline_ray/cli.py` reserves a total CPU count that is
   currently unaware of the new `num_cpu_workers` knob. If the cluster
   was sized for "one writer actor + producer" and we now dispatch K
   parallel peak-finding tasks from inside the actor, we may starve
   adjacent actors (or be starved by them) depending on what else is
   running in the pipeline.
   **Mitigation:** Task 25 must thread `num_cpu_workers` into
   `ray.init(num_cpus=…)` or into the writer actor's
   `num_cpus=num_cpu_workers + 1` reservation (see coordinator.py
   around line 244 for the current reservation pattern). Document the
   collision with the pipeline's existing Ray CPU budget.

---

## Rollback plan

If Task 26's benchmarks show `num_cpu_workers=1` regresses vs.
`aaa7cf2` (the Task 22 baseline SHA) by more than 10% on either the
light or realistic config, or if every `K > 1` row is within noise of
K=1:

1. `git checkout main` and abandon `feat/fix-bottleneck` (do not merge
   to main).
2. The test suite added in Tasks 14–19 (tests/test_peak_finding.py,
   test_writer_offline.py, test_writer_tier2.py, bench_q2_inject.py,
   plus docs/benchmarks/*.md reports) is the guardrail and stays on
   the branch as a reusable asset for the next attempt.
3. Document what went wrong in a `docs/design/axis1-attempt-<date>.md`
   retrospective on a throwaway branch before deleting
   `feat/fix-bottleneck`.
4. Revisit: either (a) vectorized peak finding (kill the Python loop
   over labeled components in `find_peaks_numpy`), or (b) GPU peak
   finding (piggyback on the model's CUDA context), or (c) algorithmic
   change (e.g. thresholded-max instead of 2-class argmax + label).
   These are bigger changes than Axis 1 and need their own design
   round.

The regression gate for Task 25 pytest (`pytest tests/ -v` green,
`num_cpu_workers=1` behavior observably equivalent to baseline) is the
earliest point at which this rollback plan would trigger — catch it
before Task 26 if possible.

---

## Scope hand-off to Task 25

Blueprint A from Task 25's description is the binding spec:

- Add `num_cpu_workers` to `config['peak_finding']` (schema placement:
  Axis-1 knob → `peak_finding` section, matching the coordinator's
  module that owns the knob — not `output`).
- Define `@ray.remote def _find_peaks_task(panel_logits, H_orig,
  W_orig, save_segmentation_maps)` in `coordinator.py`; wrap the
  existing `for panel_idx in range(logits.shape[0])` loop at
  coordinator.py:183–200.
- Plumb `num_cpu_workers` through `run_sync_pipeline` and into the
  `ray.init()` budget in `cli.py`.
- Set `OMP_NUM_THREADS=1` inside `_find_peaks_task` per Risk #3.
- Update `cxi_pipeline_ray/README.md:98`, the tuning table at
  README.md:144–146 and 222–225, and the comment block in
  `examples/configs/cxi_writer_default.yaml:44, 178, 185, 201` to
  describe the real knob (effective ceiling: `min(num_cpu_workers,
  B * C)`).
- Regression gate: `pytest tests/ -v` green; `num_cpu_workers=1`
  output byte-for-byte equivalent to baseline for the offline Tier-1/2
  tests.

Task 26 then benchmarks K ∈ {1, 2, 4, 8, 16} on the same ada node
config as Task 22 and produces the summary table / PR body.

---

## Files cited in this decision

- `docs/benchmarks/2026-04-22-baseline-light.md` lines 109–123
  (light config verdict; AMBIGUOUS / writer keeps up at 256×256,
  but peak-finding is 60.5% of what *is* spent).
- `docs/benchmarks/2026-04-22-baseline-realistic.md` lines 82, 102–127,
  132–142, 146–177, 181–194 (realistic config; 99.12% in
  `find_peaks_numpy`, 22 MB per panel, residual risks already
  enumerated).
- `docs/design/downstream-cxi-compatibility.md` lines 14–27, 114–118
  (sharding verdict SHARDED OK, stakeholder PENDING rows).
- `cxi_pipeline_ray/core/coordinator.py` lines ~183–200 (the loop to
  be replaced by Task 25).
- `examples/configs/cxi_writer_default.yaml` lines 44, 178, 185, 201
  and `cxi_pipeline_ray/README.md` lines 98, 144–146, 222–225 (docs
  to be updated by Task 25 so the knob stops lying).
