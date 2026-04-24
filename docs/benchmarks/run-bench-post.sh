#!/usr/bin/env bash
# Task 26: Run post-fix benchmark for one (config, K) pair.
# Mirrors docs/benchmarks/run-bench-profiled-v2.sh but parameterized by --writer-config.
set -euo pipefail

TAG=$1           # e.g. after-light-k1
NB=$2            # num_batches
RATE=$3          # batches_per_second
H=$4             # image height
W=$5             # image width
PEAKS=$6         # peaks_per_panel
WRITER_CFG=$7    # writer config yaml

REPO=/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
OUT=/sdf/scratch/users/c/cwang31/bench-out/${TAG}
LOG=/sdf/scratch/users/c/cwang31/bench-out/${TAG}.log

mkdir -p "${OUT}"
cd "${REPO}"
source .venv/bin/activate

python tests/bench_q2_inject.py \
  --num-batches "${NB}" \
  --batches-per-second "${RATE}" \
  --peaks-per-panel "${PEAKS}" \
  --B 1 --C 4 \
  --H-orig "${H}" --W-orig "${W}" \
  --H-preprocessed "${H}" --W-preprocessed "${W}" \
  --output-dir "${OUT}" \
  --writer-config "${WRITER_CFG}" \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 900 \
  --writer-startup-seconds 3 \
  --log-level INFO \
  2>&1 | tee "${LOG}"

echo "=== done: tag=${TAG} ==="
tail -40 "${LOG}"
