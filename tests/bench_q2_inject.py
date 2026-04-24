#!/usr/bin/env python3
"""
Tier 3 harness: Q2 injection benchmark for cxi-pipeline-ray.

This is a STANDALONE SCRIPT (not pytest). It starts a local Ray cluster,
creates a sharded Q2 queue, injects synthetic PipelineOutputs at a
controlled rate, runs ``cxi-writer`` against that queue in a background
subprocess, and reports throughput statistics.

Purpose
-------
Quantify the single-threaded bottleneck in the current writer and
provide a before/after yardstick for future Axis-1 / Axis-2
parallelization work.

Dependencies
------------
- cxi-pipeline-ray (this package) must be installed in the active venv.
- peaknet-pipeline-ray must be importable for
  ``peaknet_pipeline_ray.utils.queue.ShardedQueueManager``.
  Preferred install (from source on SDF)::

      uv pip install -e /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray

  If that repo's flat-layout breaks setuptools package discovery, append
  the following block to its ``pyproject.toml`` and retry::

      [tool.setuptools.packages.find]
      include = ["peaknet_pipeline_ray*"]

  Fallback (no install): pass ``--peaknet-pipeline-ray-path`` pointing at
  the repo root; the harness will prepend it to sys.path.

Writer config template
----------------------
The ``--writer-config`` YAML must expose the queue block the writer
connects to. A minimal template that matches the defaults used by this
harness is::

    ray:
      namespace: "peaknet-pipeline"
    queue:
      name: "peaknet_q2_bench"
      num_shards: 1
      maxsize_per_shard: 1000
      poll_timeout: 0.01
    processing:
      num_cpu_workers: 4
      max_pending_tasks: 50
    peak_finding:
      min_num_peak: 1
      max_num_peak: 2048
      connectivity: 8
    geometry:
      geom_file: null
    output:
      output_dir: "/tmp/bench-smoke"
      file_prefix: "bench"
      buffer_size: 10
      create_output_dir: true
    system:
      log_level: "INFO"
      log_file: null
      progress_interval: 50

Smoke invocation (login node)
-----------------------------
::

    python tests/bench_q2_inject.py \\
        --num-batches 20 --batches-per-second 5 \\
        --peaks-per-panel 3 --B 1 --C 4 \\
        --H-orig 128 --W-orig 128 \\
        --H-preprocessed 128 --W-preprocessed 128 \\
        --output-dir /tmp/bench-smoke \\
        --writer-config /tmp/bench-writer.yaml --seed 0
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


def _parse_args() -> argparse.Namespace:
    """Build the CLI. All tunables exposed per spec."""
    p = argparse.ArgumentParser(
        description="Q2 injection benchmark for cxi-pipeline-ray writer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Producer knobs
    p.add_argument("--num-batches", type=int, required=True,
                   help="Total number of synthetic batches to push onto Q2.")
    p.add_argument("--batches-per-second", type=float, required=True,
                   help="Target producer rate (push cadence).")
    p.add_argument("--peaks-per-panel", type=int, default=3)
    p.add_argument("--B", type=int, default=1,
                   help="Events per batch.")
    p.add_argument("--C", type=int, default=4,
                   help="Panels per event.")
    p.add_argument("--H-orig", type=int, default=1667)
    p.add_argument("--W-orig", type=int, default=1668)
    p.add_argument("--H-preprocessed", type=int, default=1696)
    p.add_argument("--W-preprocessed", type=int, default=1696)
    p.add_argument("--photon-wavelength", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)

    # Writer knobs
    p.add_argument("--output-dir", type=Path, required=True,
                   help="CXI output dir. Must match output_dir in writer config.")
    p.add_argument("--writer-config", type=Path, required=True,
                   help="Path to writer YAML consumed by cxi-writer.")
    p.add_argument("--batches-per-file", type=int, default=10)

    # Environment / plumbing
    p.add_argument("--peaknet-pipeline-ray-path", type=Path, default=None,
                   help="Optional: directory containing peaknet_pipeline_ray/ "
                        "to prepend to sys.path when the package is not installed.")
    p.add_argument("--writer-cmd", type=str, default="cxi-writer",
                   help="How to launch the writer. Override with "
                        "'python -m cxi_pipeline_ray.cli' if the console "
                        "script is not on PATH.")
    p.add_argument("--drain-poll-seconds", type=float, default=0.25,
                   help="Polling interval while waiting for the queue to drain.")
    p.add_argument("--drain-timeout-seconds", type=float, default=300.0,
                   help="Hard timeout on the drain phase; harness aborts after.")
    p.add_argument("--writer-startup-seconds", type=float, default=5.0,
                   help="Grace period before starting to push so the writer "
                        "has time to connect to Ray and the queue.")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("bench_q2_inject")


def _load_writer_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_sharded_queue_manager(
    extra_path: Optional[Path],
    logger: logging.Logger,
):
    """
    Return the ``ShardedQueueManager`` class, loaded by whichever path works.

    Resolution order:

    1. If ``peaknet_pipeline_ray`` is already installed, import normally.
    2. Otherwise, load ``peaknet_pipeline_ray/utils/queue.py`` as a
       standalone module via importlib. This sidesteps the package's
       ``__init__.py`` which eagerly imports torch and hydra — neither of
       which are deps of cxi-pipeline-ray. The queue module itself only
       uses ray + stdlib and is safe to load in isolation.

    Returns:
        The ``ShardedQueueManager`` class.
    """
    try:
        from peaknet_pipeline_ray.utils.queue import ShardedQueueManager
        logger.debug("peaknet_pipeline_ray.utils.queue is already installed.")
        return ShardedQueueManager
    except ImportError:
        pass

    if extra_path is None:
        raise RuntimeError(
            "peaknet_pipeline_ray is not installed. Either install it "
            "(`uv pip install -e /path/to/peaknet-pipeline-ray`) or pass "
            "--peaknet-pipeline-ray-path pointing at the repo root."
        )

    queue_py = extra_path.resolve() / "peaknet_pipeline_ray" / "utils" / "queue.py"
    if not queue_py.is_file():
        raise RuntimeError(f"Could not find queue.py at {queue_py}.")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "peaknet_pipeline_ray_queue_shim", queue_py,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"importlib.util failed to build a spec for {queue_py}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    logger.info("Loaded ShardedQueueManager directly from %s.", queue_py)
    return module.ShardedQueueManager


def _build_producer_batches(args: argparse.Namespace, logger: logging.Logger) -> List:
    """
    Pre-build all synthetic PipelineOutputs before timing begins.

    Why pre-build: the fixture uses ray.put() which is not free. By
    building up front, the producer loop only measures queue-put latency
    and inter-push pacing, not fixture construction time.
    """
    # Ensure the tests/ package from this repo is importable no matter which
    # cwd the harness is launched from.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.fixtures import make_synthetic_pipeline_output

    batches = []
    for i in range(args.num_batches):
        fake, _gt = make_synthetic_pipeline_output(
            B=args.B,
            C=args.C,
            H_orig=args.H_orig,
            W_orig=args.W_orig,
            H_preprocessed=args.H_preprocessed,
            W_preprocessed=args.W_preprocessed,
            peaks_per_panel=args.peaks_per_panel,
            photon_wavelength=args.photon_wavelength,
            timestamp=(1 << 32) | (i + 1),  # nonzero so writer emits timestamp dataset
            seed=args.seed + i,
            draw_peaks_in_image=False,  # saves cycles during build
        )
        batches.append(fake)
    logger.info("Pre-built %d synthetic batches.", len(batches))
    return batches


_WRITER_BOOTSTRAP_TEMPLATE = '''\
"""Auto-generated writer bootstrap — installs a minimal peaknet_pipeline_ray
shim so cxi-writer can import ShardedQueueManager without pulling in torch
or hydra, then calls cxi_pipeline_ray.cli.main().
"""
import importlib.util
import sys
import types
from pathlib import Path

QUEUE_PY = Path({queue_py!r})
REPO_ROOT = Path({repo_root!r})

# Make sure the cxi-pipeline-ray repo root is importable — needed so the
# pickled FakePipelineOutput objects (class defined in tests/fixtures.py)
# can be deserialized inside the writer process.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Build a synthetic peaknet_pipeline_ray package with only utils.queue populated.
pkg = types.ModuleType("peaknet_pipeline_ray")
pkg.__path__ = []
utils = types.ModuleType("peaknet_pipeline_ray.utils")
utils.__path__ = []
pkg.utils = utils
sys.modules["peaknet_pipeline_ray"] = pkg
sys.modules["peaknet_pipeline_ray.utils"] = utils

spec = importlib.util.spec_from_file_location(
    "peaknet_pipeline_ray.utils.queue", str(QUEUE_PY),
)
mod = importlib.util.module_from_spec(spec)
sys.modules["peaknet_pipeline_ray.utils.queue"] = mod
utils.queue = mod
spec.loader.exec_module(mod)

from cxi_pipeline_ray.cli import main
sys.exit(main())
'''


def _write_writer_bootstrap(extra_path: Path, dest: Path, logger: logging.Logger) -> Path:
    """Emit a small Python wrapper that shims in peaknet_pipeline_ray.utils.queue
    and then invokes cxi-writer's main. Returns the bootstrap path.
    """
    queue_py = (extra_path.resolve() / "peaknet_pipeline_ray" / "utils" / "queue.py")
    repo_root = Path(__file__).resolve().parent.parent
    script = _WRITER_BOOTSTRAP_TEMPLATE.format(
        queue_py=str(queue_py),
        repo_root=str(repo_root),
    )
    dest.write_text(script)
    logger.info("Wrote writer bootstrap to %s", dest)
    return dest


def _start_writer_subprocess(
    args: argparse.Namespace,
    logger: logging.Logger,
    ray_address: str,
) -> subprocess.Popen:
    """Launch cxi-writer as a child process; stdout/stderr flow to ours.

    Behavior:

    - If ``--writer-cmd`` was overridden, invoke it verbatim (legacy path).
    - Otherwise, if ``--peaknet-pipeline-ray-path`` was given, emit a small
      Python bootstrap that installs the queue-module shim and then calls
      cxi_pipeline_ray.cli.main(). This keeps the subprocess architecture
      the spec asks for while letting us run without torch in the venv.
    - Otherwise, fall through to ``cxi-writer`` (the installed console script).

    The child inherits ``RAY_ADDRESS`` so its ``ray.init()`` attaches to
    the bench's Ray head (same namespace) instead of spinning up a
    second, isolated cluster.
    """
    default_cmd = args.writer_cmd == "cxi-writer"
    if default_cmd and args.peaknet_pipeline_ray_path is not None:
        bootstrap = _write_writer_bootstrap(
            args.peaknet_pipeline_ray_path,
            Path("/tmp") / f"bench_writer_bootstrap_{os.getpid()}.py",
            logger,
        )
        writer_cmd_parts = [sys.executable, str(bootstrap)]
    else:
        writer_cmd_parts = args.writer_cmd.split()

    cmd_parts = writer_cmd_parts + [
        "--config", str(args.writer_config),
        "--batches-per-file", str(args.batches_per_file),
        "--output-dir", str(args.output_dir),
        "--log-level", args.log_level,
    ]
    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address
    logger.info("Launching writer subprocess (RAY_ADDRESS=%s): %s",
                ray_address, " ".join(cmd_parts))
    return subprocess.Popen(
        cmd_parts,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
        preexec_fn=os.setsid,  # so we can SIGTERM the whole group
    )


def _percentile(values: List[float], pct: float) -> float:
    """Small dependency-free percentile helper."""
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def _count_cxi_events(output_dir: Path) -> tuple:
    """Scan .cxi output files and return (num_files, total_events)."""
    try:
        import h5py  # type: ignore
    except ImportError:
        return (0, 0)

    cxi_files = sorted(output_dir.glob("*.cxi"))
    total = 0
    for path in cxi_files:
        try:
            with h5py.File(path, "r") as f:
                if "/entry_1/result_1/nPeaks" in f:
                    total += int(f["/entry_1/result_1/nPeaks"].shape[0])
        except Exception:
            continue
    return (len(cxi_files), total)


def main() -> int:
    args = _parse_args()
    logger = _configure_logging(args.log_level)

    # Resolve ShardedQueueManager before anything else touches ray.
    ShardedQueueManager = _load_sharded_queue_manager(
        args.peaknet_pipeline_ray_path, logger,
    )

    import ray  # ray must be imported after any sys.path tweaks

    writer_cfg = _load_writer_config(args.writer_config)
    queue_cfg = writer_cfg["queue"]
    ray_cfg = writer_cfg.get("ray", {})
    namespace = ray_cfg.get("namespace", "peaknet-pipeline")

    # Make sure output dir exists (matches create_output_dir: true semantics).
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing Ray in namespace=%s ...", namespace)
    ctx = ray.init(namespace=namespace, ignore_reinit_error=True, include_dashboard=False)
    # ``address`` is the GCS address the child's RAY_ADDRESS=auto would
    # discover, but being explicit keeps the wiring reliable even when
    # /tmp/ray/ray_current_cluster isn't freshly written.
    ray_address = getattr(ctx, "address_info", {}).get("gcs_address", "auto")
    logger.info("Ray GCS address: %s", ray_address)

    # Q2: create the queue BEFORE the writer starts so it attaches to
    # the existing detached actors instead of creating duplicates.
    q2 = ShardedQueueManager(
        base_name=queue_cfg["name"],
        num_shards=queue_cfg["num_shards"],
        maxsize_per_shard=queue_cfg.get("maxsize_per_shard", 1000),
    )
    logger.info(
        "Created ShardedQueueManager(name=%s, num_shards=%s, maxsize_per_shard=%s).",
        queue_cfg["name"], queue_cfg["num_shards"], queue_cfg.get("maxsize_per_shard", 1000),
    )

    # Pre-build batches so the producer loop is only pacing + put().
    batches = _build_producer_batches(args, logger)

    # Start the writer subprocess. It will attach to the same detached
    # Q2 actors above and start its sync-pipeline loop.
    writer_proc = _start_writer_subprocess(args, logger, ray_address)

    if args.writer_startup_seconds > 0:
        logger.info("Sleeping %.2fs to let writer connect ...", args.writer_startup_seconds)
        time.sleep(args.writer_startup_seconds)

    # ---- Producer loop ------------------------------------------------
    target_interval = 1.0 / args.batches_per_second if args.batches_per_second > 0 else 0.0
    push_timestamps: List[float] = []
    queue_depths: List[int] = []

    logger.info(
        "Pushing %d batches at target rate %.2f batches/sec ...",
        args.num_batches, args.batches_per_second,
    )
    producer_start = time.perf_counter()
    for i, batch in enumerate(batches):
        push_start = time.perf_counter()
        ok = q2.put(batch)
        if not ok:
            logger.warning(
                "Queue full at batch %d; backing off briefly and retrying.", i,
            )
            # Simple retry with a short sleep — prevents tight busy-wait.
            while not q2.put(batch):
                time.sleep(0.05)
        push_timestamps.append(push_start)
        queue_depths.append(q2.size())

        # Pace to target rate.
        if target_interval > 0:
            next_push_at = producer_start + (i + 1) * target_interval
            sleep_for = next_push_at - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    producer_end = time.perf_counter()
    logger.info(
        "Producer pushed %d batches in %.3fs (%.2f batches/sec achieved).",
        args.num_batches,
        producer_end - producer_start,
        args.num_batches / max(producer_end - producer_start, 1e-9),
    )

    # ---- Drain ---------------------------------------------------------
    logger.info("Waiting for queue to drain (timeout=%.1fs) ...", args.drain_timeout_seconds)
    drain_start = time.perf_counter()
    last_depth = q2.size()
    while True:
        depth = q2.size()
        if depth == 0:
            break
        if time.perf_counter() - drain_start > args.drain_timeout_seconds:
            logger.error("Drain timeout — queue still has %d items.", depth)
            break
        if depth != last_depth:
            logger.debug("Queue depth: %d", depth)
            last_depth = depth
        time.sleep(args.drain_poll_seconds)
    drain_end = time.perf_counter()

    # Give the writer a moment to flush the final CXI file, then stop it.
    time.sleep(2.0)
    logger.info("Sending SIGTERM to writer process group (pid=%s).", writer_proc.pid)
    try:
        os.killpg(os.getpgid(writer_proc.pid), signal.SIGTERM)
        writer_proc.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        logger.warning("Writer did not stop on SIGTERM; sending SIGKILL.")
        os.killpg(os.getpgid(writer_proc.pid), signal.SIGKILL)
        writer_proc.wait(timeout=10.0)
    except ProcessLookupError:
        pass  # already exited

    # ---- Report --------------------------------------------------------
    total_files, total_events = _count_cxi_events(args.output_dir)

    inter_push = [
        push_timestamps[i] - push_timestamps[i - 1]
        for i in range(1, len(push_timestamps))
    ]
    achieved_rate = args.num_batches / max(producer_end - producer_start, 1e-9)
    end_to_end_seconds = drain_end - producer_start

    print("\n=== bench_q2_inject report ===")
    print(f"num_batches_pushed           : {args.num_batches}")
    print(f"batches_per_second_target    : {args.batches_per_second:.3f}")
    print(f"batches_per_second_achieved  : {achieved_rate:.3f}")
    print(f"producer_wall_seconds        : {producer_end - producer_start:.3f}")
    print(f"drain_wall_seconds           : {drain_end - producer_end:.3f}")
    print(f"end_to_end_wall_seconds      : {end_to_end_seconds:.3f}")
    print(f"inter_push_p50_seconds       : {_percentile(inter_push, 50):.6f}")
    print(f"inter_push_p95_seconds       : {_percentile(inter_push, 95):.6f}")
    print(f"inter_push_p99_seconds       : {_percentile(inter_push, 99):.6f}")
    print(f"queue_depth_max              : {max(queue_depths) if queue_depths else 0}")
    print(f"cxi_files_written            : {total_files}")
    print(f"cxi_events_written           : {total_events}")
    print(f"events_per_batch_expected    : {args.B}")
    print(f"events_total_expected        : {args.B * args.num_batches}")
    print("=== end report ===")

    ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
