#!/usr/bin/env bash
# Run bench_q2_inject.py + attach py-spy to the writer subprocess.
# Usage: run-bench-profiled.sh <tag> <num_batches> <rate> <H> <W> <peaks>
set -euo pipefail

TAG=$1
NB=$2
RATE=$3
H=$4
W=$5
PEAKS=$6

REPO=/sdf/data/lcls/ds/prj/prjcwang31/results/codes/cxi-pipeline-ray
OUT=/tmp/bench-out/${TAG}
LOG=/tmp/bench-out/${TAG}.log
PROF=/tmp/bench-out/${TAG}.speedscope.json

mkdir -p "${OUT}"
cd "${REPO}"
source .venv/bin/activate

# Launch bench in background; log to file.
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
  --log-level INFO \
  > "${LOG}" 2>&1 &
BENCH_PID=$!

# Wait for writer subprocess to appear.
WRITER_PID=""
for i in $(seq 1 40); do
  sleep 0.5
  WRITER_PID=$(grep -oE 'Launching writer subprocess.*pgrep|Launching writer subprocess \(RAY' "${LOG}" 2>/dev/null | head -1 || true)
  # Safer: parse "RAY_ADDRESS=... " line + then locate the child PID via pstree
  CAND=$(pgrep -P ${BENCH_PID} -f 'bench_writer_bootstrap' 2>/dev/null | head -1 || true)
  if [[ -n "${CAND}" ]]; then
    WRITER_PID="${CAND}"
    break
  fi
done

echo "bench_pid=${BENCH_PID} writer_pid=${WRITER_PID:-NONE}"

if [[ -n "${WRITER_PID}" ]]; then
  # Give the writer a couple of seconds to start processing before sampling.
  sleep 2
  /sdf/home/c/cwang31/.local/bin/py-spy record \
    --pid "${WRITER_PID}" \
    --duration 25 \
    --rate 200 \
    --subprocesses \
    --format speedscope \
    --output "${PROF}" \
    > /tmp/bench-out/${TAG}.pyspy.log 2>&1 &
  PYSPY_PID=$!
  echo "pyspy_pid=${PYSPY_PID} output=${PROF}"
fi

wait ${BENCH_PID}
BENCH_RC=$?
echo "bench_rc=${BENCH_RC}"

# Make sure py-spy has finished (duration may outlast bench for short runs).
wait ${PYSPY_PID:-} 2>/dev/null || true

echo "=== bench report tail ==="
tail -40 "${LOG}"
