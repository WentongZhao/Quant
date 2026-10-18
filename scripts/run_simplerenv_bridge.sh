#!/usr/bin/env bash
# Usage: bash scripts/run_simplerenv_bridge.sh CHECKPOINT [OUTPUT_DIR]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CHECKPOINT=${1:?Supply the OpenVLA checkpoint directory or model ID}
OUTPUT=${2:-"$ROOT/runs/$(date +%Y%m%d-%H%M%S)-openvla-bridge"}
mkdir -p "$OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd)
PYTHON=${SIMPLERENV_PYTHON:-python}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg TOKENIZERS_PARALLELISM=false
trap 'echo $? > "$OUTPUT/exitcode"' EXIT
cd "$ROOT"
for phase in smoke full; do
    CONFIG="$ROOT/configs/evaluation/simplerenv_bridge.json"
    if [[ "$phase" == smoke ]]; then CONFIG="$ROOT/configs/evaluation/simplerenv_bridge_smoke.json"; fi
    "$PYTHON" evaluate.py --model_backend openvla --benchmark simplerenv --mode fp         --checkpoint "$CHECKPOINT" --tasks_json "$CONFIG" --output_dir "$OUTPUT/$phase"         > "$OUTPUT/$phase.log" 2>&1
    "$PYTHON" - "$OUTPUT/$phase/results.json" "$phase" <<'CHECK'
import json, sys
r = json.load(open(sys.argv[1]))
expected = 4 if sys.argv[2] == "smoke" else 96
assert r["status"] == "completed" and r["total_episodes"] == expected, r
print(sys.argv[2], r["total_successes"], "/", r["total_episodes"], flush=True)
CHECK
done
