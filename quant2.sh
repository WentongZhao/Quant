#!/usr/bin/env bash
set -euo pipefail

# SmoothQuant (vision-only / timm ViT) sweep for LIBERO:
# 4 task suites (libero_10/libero_spatial/libero_object/libero_goal)
# x 2 settings (w4a4kv16, w8a8kv16)

ICML_DIR=/file02/user/liyuye/quantization_VLA/openvla-oft-main/gamma_patch

# Fixed GPU as requested
export CUDA_VISIBLE_DEVICES=4

CALIB_SAMPLING=proprio_stratified
CALIB_TAG=proprio_stratified
N_SAMPLES=256
SEED=7

# SmoothQuant knobs
SQ_ALPHA=0.5
SQ_SCALE_MODE=max

# Keep your base path style; mark this run as vision-only SQ explicitly.
LOG_BASE=${ICML_DIR}/smoothquant_vit
SCALE_BASE=${ICML_DIR}/scales/smoothquant_vit

run_one_setting() {
    local task="$1"
    local model="$2"
    local data="$3"
    local w_bit="$4"
    local a_bit="$5"
    local cfg_tag="vit_w${w_bit}a${a_bit}kv16"

    local scale_dir="${SCALE_BASE}/${task}/${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}"
    mkdir -p "${scale_dir}" "${LOG_BASE}/${task}"

    python main_quant.py \
        --model "${model}" \
        --data "${data}" \
        --method smoothquant \
        --sq_target vision \
        --sq_a_asym --no-sq_w_asym \
        --alpha "${SQ_ALPHA}" \
        --sq_scale_mode "${SQ_SCALE_MODE}" \
        --w_bit "${w_bit}" --a_bit "${a_bit}" --k_bit 16 --v_bit 16 \
        --sq_vision_scale_path "${scale_dir}/vision_scale.pt" \
        --run_process \
        --deterministic \
        --calib_sampling "${CALIB_SAMPLING}" \
        --calib_prompt_balance \
        --n_samples "${N_SAMPLES}" \
        --task_suite_name "${task}" \
        --seed "${SEED}" 2>&1 | tee "${LOG_BASE}/${task}/smoothquant_vit_${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}_${N_SAMPLES}_scale.log"

    python main_quant.py \
        --model "${model}" \
        --data "${data}" \
        --method smoothquant \
        --sq_target vision \
        --sq_a_asym --no-sq_w_asym \
        --alpha "${SQ_ALPHA}" \
        --sq_scale_mode "${SQ_SCALE_MODE}" \
        --w_bit "${w_bit}" --a_bit "${a_bit}" --k_bit 16 --v_bit 16 \
        --sq_vision_scale_path "${scale_dir}/vision_scale.pt" \
        --pseudo_quant \
        --deterministic \
        --calib_sampling "${CALIB_SAMPLING}" \
        --calib_prompt_balance \
        --n_samples "${N_SAMPLES}" \
        --eval_libero \
        --task_suite_name "${task}" \
        --num_trials_per_task 50 \
        --seed "${SEED}" 2>&1 | tee "${LOG_BASE}/${task}/smoothquant_vit_${cfg_tag}_alpha${SQ_ALPHA}_${CALIB_TAG}_${N_SAMPLES}_eval.log"
}

run_suite() {
    local task="$1"
    local model="$2"
    local data="$3"

    run_one_setting "${task}" "${model}" "${data}" 8 8
    run_one_setting "${task}" "${model}" "${data}" 4 4
}

run_suite \
    "libero_10" \
    "/file02/user/liyuye/models/openvla-7b-oft-finetuned-libero-10" \
    "/file02/user/liyuye/dataset/modified_libero_rlds/libero_10_no_noops/1.0.0"

run_suite \
    "libero_spatial" \
    "/file02/user/liyuye/models/openvla-7b-oft-finetuned-libero-spatial" \
    "/file02/user/liyuye/dataset/modified_libero_rlds/libero_spatial_no_noops/1.0.0"

run_suite \
    "libero_object" \
    "/file02/user/liyuye/models/openvla-7b-oft-finetuned-libero-object" \
    "/file02/user/liyuye/dataset/modified_libero_rlds/libero_object_no_noops/1.0.0"

run_suite \
    "libero_goal" \
    "/file02/user/liyuye/models/openvla-7b-oft-finetuned-libero-goal" \
    "/file02/user/liyuye/dataset/modified_libero_rlds/libero_goal_no_noops/1.0.0"