#!/usr/bin/env bash
set -euo pipefail
QUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCREEN_NAME="${SCREEN_NAME:-libero-unified-$(date +%Y%m%d-%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-${QUANT_ROOT}/runs/$(date +%Y%m%d-%H%M%S)-libero-comparison}"
if [[ -e "${RUN_ROOT}" ]]; then
    echo "Choose a new RUN_ROOT; already exists: ${RUN_ROOT}" >&2
    exit 1
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd -- "${RUN_ROOT}" && pwd)"
export RUN_ROOT
printf '%s\n' "${SCREEN_NAME}" > "${RUN_ROOT}/screen-name.txt"
screen -dmS "${SCREEN_NAME}" bash "${QUANT_ROOT}/scripts/run_libero_comparison.sh"
printf 'screen: %s\noutput: %s\nattach: screen -r %s\n' "${SCREEN_NAME}" "${RUN_ROOT}" "${SCREEN_NAME}"
