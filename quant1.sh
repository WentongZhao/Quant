#!/usr/bin/env bash
set -euo pipefail

# SmoothQuant sweep for LIBERO:
# 4 task suites (libero_10/libero_spatial/libero_object/libero_goal)
# x 2 settings (w4a4kv16, w8a8kv16)

# 项目根目录（用于保存日志与scale）
ICML_DIR=/root/autodl-tmp/Quant

# Fixed GPU as requested
export CUDA_VISIBLE_DEVICES=0

CALIB_SAMPLING=proprio_stratified
CALIB_TAG=proprio_stratified
N_SAMPLES=256
SEED=7

# SmoothQuant knobs
SQ_ALPHA=0.5

SQ_SCALE_MODE=max

LOG_BASE=${ICML_DIR}/smoothquant_logs
SCALE_BASE=${ICML_DIR}/smoothquant_scales

run_one_setting() {
    local task="$1"
    local model="$2"
    local data="$3"
    local w_bit="$4"
    local a_bit="$5"
    local cfg_tag="w${w_bit}a${a_bit}kv16"

    local scale_dir="${SCALE_BASE}/${task}/${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}"
    mkdir -p "${scale_dir}" "${LOG_BASE}/${task}"

    # 1) 生成 scale（权重对称、激活非对称）
    python main_quant.py \
        --model "${model}" \
        --data "${data}" \
        --method smoothquant \
        --sq_a_asym --no-sq_w_asym \
        --alpha "${SQ_ALPHA}" \
        --sq_scale_mode "${SQ_SCALE_MODE}" \
        --w_bit "${w_bit}" --a_bit "${a_bit}" --k_bit 16 --v_bit 16 \
        --scale_path "${scale_dir}/scale.pt" \
        --run_process \
        --deterministic \
        --calib_sampling "${CALIB_SAMPLING}" \
        --calib_prompt_balance \
        --n_samples "${N_SAMPLES}" \
        --task_suite_name "${task}" \
        --seed "${SEED}" |& tee "${LOG_BASE}/${task}/smoothquant_${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}_${N_SAMPLES}_scale.log"

    # 2) pseudo quant + eval（权重对称、激活非对称）
    python main_quant.py \
        --model "${model}" \
        --data "${data}" \
        --method smoothquant \
        --sq_a_asym --no-sq_w_asym \
        --alpha "${SQ_ALPHA}" \
        --sq_scale_mode "${SQ_SCALE_MODE}" \
        --w_bit "${w_bit}" --a_bit "${a_bit}" --k_bit 16 --v_bit 16 \
        --scale_path "${scale_dir}/scale.pt" \
        --pseudo_quant \
        --deterministic \
        --calib_sampling "${CALIB_SAMPLING}" \
        --calib_prompt_balance \
        --n_samples "${N_SAMPLES}" \
        --eval_libero \
        --task_suite_name "${task}" \
        --num_trials_per_task 50 \
        --seed "${SEED}" |& tee "${LOG_BASE}/${task}/smoothquant_${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}_${N_SAMPLES}_eval.log"
}

run_suite() {
    local task="$1"
    local model="$2"
    local data="$3"

    run_one_setting "${task}" "${model}" "${data}" 4 4
    run_one_setting "${task}" "${model}" "${data}" 8 8
}

# =========================
# ✅ 只跑 libero_spatial
# =========================
run_suite \
  "libero_spatial" \
  "/root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-spatial" \
  "/root/autodl-tmp/libero_spatial_no_noops/1.0.0"

# =========================
# 其它 suite（暂不跑，已安全注释）
# =========================
: <<'BLOCK_LIBERO_10'
run_suite \
  "libero_10" \
  "/root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-10" \
  "/root/autodl-tmp/libero_10_no_noops/1.0.0"
BLOCK_LIBERO_10

: <<'BLOCK_LIBERO_OBJECT'
run_suite \
  "libero_object" \
  "/root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-object" \
  "/root/autodl-tmp/libero_object_no_noops/1.0.0"
BLOCK_LIBERO_OBJECT

: <<'BLOCK_LIBERO_GOAL'
run_suite \
  "libero_goal" \
  "/root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-goal" \
  "/root/autodl-tmp/libero_goal_no_noops/1.0.0"
BLOCK_LIBERO_GOAL