# Baseline Benchmark — Light Config (256x256)

**Task:** 22 — quantify where the single-threaded bottleneck in the CXI writer lives.
**Status:** Baseline, UNMODIFIED code. No Axis-1/2 changes applied yet.
**Companion report:** `2026-04-22-baseline-realistic.md` (production shapes).

---

## Command

```bash
docs/benchmarks/run-bench-profiled-v2.sh baseline-light 200 20 256 256 3
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
  --output-dir /sdf/scratch/users/c/cwang31/bench-out/baseline-light \
  --writer-config docs/benchmarks/bench-writer.yaml \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 600 \
  --writer-startup-seconds 3 \
  --log-level INFO
```

py-spy was attached to the writer subprocess for the middle ~25s of the run
(`rate=200`, `--subprocesses`, speedscope output).

## Code under test

- Repo: `cxi-pipeline-ray`
- Branch: `feat/fix-bottleneck`
- Git SHA: `aaa7cf2` (`aaa7cf2862cbaffc0a87e5a62e81cf6a24f08aee`) — tip of
  `feat/fix-bottleneck` at run time, identical writer code to `main` (no fix
  applied yet; branch only adds test fixtures from Tasks 14–19).

## Machine

- Host: `sdfada002` (SLAC SDF, `ada` partition)
- CPU: AMD EPYC 7713P 64-Core (96 logical CPUs exposed to the node; Ray
  reported `CPU: 96.0` to the cluster)
- GPU: 4× NVIDIA L40S, 46 GB — **not used** by this benchmark
- RAM: 703 GB (288 GB allocated to the job)
- OS: RHEL 8.6, kernel 4.18.0-372.32.1
- Allocation: `salloc --no-shell --account=lcls:prjdat21 --partition=ada
  --nodes=1 --gpus=4 --time=4:00:00` (JOBID 25663264)
- Node CPU alloc: 72 CPUs (Ray saw 96 because `include_dashboard=False` and
  Ray measures the host, not the cgroup — noted for report consistency)
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
inter_push_p95_seconds       : 0.050012
inter_push_p99_seconds       : 0.067042
queue_depth_max              : 4
cxi_files_written            : 20
cxi_events_written           : 200
events_per_batch_expected    : 1
events_total_expected        : 200
```

### Per-batch processing (writer side, parsed from coordinator log timestamps)

The coordinator log only stamps at 1 s granularity, so percentile numbers from
it are too coarse to quote as p95/p99; they are included for completeness.

- n = 199 inter-batch deltas (batches 2..200)
- mean = 50.3 ms (= 1 / 20 bps; writer keeps up perfectly with producer)
- effective sustained rate = 200 batches / 10.002 s ≈ **20.0 batches/sec** (matches target)

## py-spy dominant-stage numbers

File: `baseline-light.speedscope.json`, total weight 2.685 s of sampled
writer CPU time (3 profiles — main + Ray worker threads).

**Top by SELF time (where the CPU *was*):**

| % self | Function | File |
|---|---|---|
| 27.93% | `numpy._wrapfunc` | called from `np.argmax` over logits |
| 16.57% | `find_peaks_numpy` (leaf at peak_finding.py:41) | center-of-mass inner loop |
| 16.39% | `ray._private.worker.get_objects` | pulling preprocessed batch via Ray object store |
| 10.24% | `scipy.ndimage._measurements.label` | 8-connectivity component labeling |
|  5.77% | `ray.actor._actor_method_call` | submit to writer actor |

**Top by TOTAL time (paths through which time flowed):**

- `process_batch` (coordinator.py) = **69.27%**
- `find_peaks_numpy` = **60.52%**
- `ray.get_objects` path = 24.58% (Q2 dequeue)
- `ray.actor._remote` / `_actor_method_call` = 10.43% (submit to writer)

## Verdict

**AMBIGUOUS / WRITER KEEPS UP at 256x256.**

The writer never falls behind a 20 batches/sec producer at this image size
— drain wall is 2 ms, queue depth maxes at 4. The bottleneck cannot be
characterized from this config alone because the system is **not under
backpressure**. What we *can* say from the py-spy trace is that the work
that *does* happen in `process_batch` is dominated by `find_peaks_numpy`
(60.5% of total time) and specifically by its numpy inner loop
(`argmax` 27.9% + center-of-mass 16.6% + `scipy.ndimage.label` 10.2%).
CXI flush and HDF5 writes do not register above noise. Extrapolating this
mix to production shapes predicts peak-finding dominance, but the
extrapolation must be confirmed — that is the point of
`baseline-realistic.md`.

**Verdict:** **PEAK FINDING DOMINATES THE WORK, BUT THROUGHPUT IS NOT
BOUND at this scale.** Use the realistic config report
(`baseline-realistic.md`) for the axis decision.

## Artifacts

- Log: `/sdf/scratch/users/c/cwang31/bench-out/baseline-light.log`
- py-spy: `/sdf/scratch/users/c/cwang31/bench-out/baseline-light.speedscope.json`
- CXI output: `/sdf/scratch/users/c/cwang31/bench-out/baseline-light/*.cxi`
  (20 files, ~10 MB each — 200 events total)
- Parser: `docs/benchmarks/parse-pyspy.py`
