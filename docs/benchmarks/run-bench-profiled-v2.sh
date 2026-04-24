#!/usr/bin/env bash
# Run bench_q2_inject.py + attach py-spy to the writer subprocess.
# v2: write CXI output to /sdf/scratch to avoid node-local /tmp exhaustion
# at 1696x1696. Longer writer-PID detection window.
set -euo pipefail

TAG=$1
NB=$2
RATE=$3
H=$4
W=$5
PEAKS=$6

REPO=/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
OUT=/sdf/scratch/users/c/cwang31/bench-out/${TAG}
LOG=/sdf/scratch/users/c/cwang31/bench-out/${TAG}.log
PROF=/sdf/scratch/users/c/cwang31/bench-out/${TAG}.speedscope.json

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
  --writer-config "${REPO}/docs/benchmarks/bench-writer.yaml" \
  --peaknet-pipeline-ray-path /sdf/data/lcls/ds/prj/prjcwang31/results/codes/peaknet-pipeline-ray \
  --seed 0 \
  --drain-timeout-seconds 600 \
  --writer-startup-seconds 3 \
  --log-level INFO \
  > "${LOG}" 2>&1 &
BENCH_PID=$!

# Wait up to 90 seconds for the writer subprocess to appear (needed because
# 1696x1696 pre-build of 500 batches is slow; writer bootstrap spawns
# *after* the pre-build in our harness flow).
WRITER_PID=""
for i in $(seq 1 180); do
  sleep 0.5
  CAND=$(pgrep -P "${BENCH_PID}" -f 'bench_writer_bootstrap' 2>/dev/null | head -1 || true)
  if [[ -n "${CAND}" ]]; then
    WRITER_PID="${CAND}"
    break
  fi
  # Also check whether bench has already exited (abort early).
  if ! kill -0 "${BENCH_PID}" 2>/dev/null; then
    break
  fi
done

echo "bench_pid=${BENCH_PID} writer_pid=${WRITER_PID:-NONE}"

if [[ -n "${WRITER_PID}" ]]; then
  sleep 3  # let writer get into steady state
  /sdf/home/c/cwang31/.local/bin/py-spy record \
    --pid "${WRITER_PID}" \
    --duration 25 \
    --rate 200 \
    --subprocesses \
    --format speedscope \
    --output "${PROF}" \
    > /sdf/scratch/users/c/cwang31/bench-out/${TAG}.pyspy.log 2>&1 &
  PYSPY_PID=$!
  echo "pyspy_pid=${PYSPY_PID} output=${PROF}"
fi

wait ${BENCH_PID}
BENCH_RC=$?
echo "bench_rc=${BENCH_RC}"

wait ${PYSPY_PID:-} 2>/dev/null || true

echo "=== bench report tail ==="
tail -40 "${LOG}"
