#!/usr/bin/env bash
set -euo pipefail

QUANT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${QUANT_PYTHON:-/root/miniconda3/envs/quant-openvla-oft/bin/python}"
RUN_ROOT="${RUN_ROOT:-${QUANT_ROOT}/runs/$(date +%Y%m%d-%H%M%S)-libero-comparison}"
NUM_TRIALS="${NUM_TRIALS:-50}"
CASES="${CASES:-fp w8a8 w4a4}"
mkdir -p "${RUN_ROOT}" "${QUANT_ROOT}/runs"
RUN_ROOT="$(cd -- "${RUN_ROOT}" && pwd)"
exec 9>"${QUANT_ROOT}/runs/.libero-comparison.lock"
flock -n 9 || { echo 'Another LIBERO comparison is already running.'; exit 1; }
exec > >(tee -a "${RUN_ROOT}/driver.log") 2>&1
trap 'code=$?; printf "%s\n" "$code" > "${RUN_ROOT}/exitcode"; echo "[driver] exit=$code at $(date -Is)"' EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export TF_CPP_MIN_LOG_LEVEL=3 TOKENIZERS_PARALLELISM=false
export TZ=Asia/Shanghai

echo "[driver] started=$(date -Is) cases=${CASES} trials_per_task=${NUM_TRIALS}"
echo "[driver] root=${RUN_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "${RUN_ROOT}/gpu.csv"
"${PYTHON}" -m pip freeze > "${RUN_ROOT}/packages.txt"
task_args=()
if [[ -n "${EVAL_TASK_IDS:-}" ]]; then
    read -ra task_ids <<< "${EVAL_TASK_IDS}"
    task_args=(--eval_task_ids "${task_ids[@]}")
fi
read -ra case_list <<< "${CASES}"
for case_name in "${case_list[@]}"; do
    args=(--config "${QUANT_ROOT}/configs/evaluation/libero_spatial.yaml"
          --num_trials_per_task "${NUM_TRIALS}" --output_dir "${RUN_ROOT}/${case_name}")
    case "${case_name}" in
        fp) args+=(--mode fp) ;;
        w8a8|w4a4)
            bits="${case_name:1:1}"
            scale="${QUANT_ROOT}/smoothquant_scales/libero_spatial/${case_name}kv16_alpha0.5_proprio_stratified/scale.pt"
            args+=(--mode quant --w_bit "${bits}" --a_bit "${bits}" --scale_path "${scale}")
            ;;
        *) echo "Unknown case: ${case_name}"; exit 2 ;;
    esac
    echo "[driver] case=${case_name} started=$(date -Is)"
    "${PYTHON}" -u "${QUANT_ROOT}/evaluate.py" "${args[@]}" "${task_args[@]}" \
        2>&1 | tee "${RUN_ROOT}/${case_name}.console.log"
    echo "[driver] case=${case_name} completed=$(date -Is)"
done
