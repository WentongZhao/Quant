#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NAME=${SIMPLERENV_SCREEN_NAME:-openvla-bridge-$(date +%Y%m%d-%H%M%S)}
command -v screen >/dev/null
screen -dmS "$NAME" bash "$ROOT/scripts/run_simplerenv_bridge.sh" "$@"
echo "Started screen: $NAME"
echo "Attach: screen -r $NAME"
