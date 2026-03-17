import argparse
import hashlib
import os
import warnings
from typing import Union

import numpy as np
import tqdm
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

import torch

from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM, IGNORE_INDEX, STOP_INDEX
from experiments.robot.libero.run_libero_eval import (
    GenerateConfig,
    validate_config,
    setup_logging,
    load_initial_states,
    run_task,
    check_unnorm_key,
)
from experiments.robot.openvla_utils import get_vla, get_processor, get_proprio_projector, get_action_head, get_noisy_action_projector
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere
from qvla.calibration.get_libero import get_libero_calib_dataset
from qvla.quantization.quant_wrapper import qwrapper
try:
    from qvla.methods.mbq.quantize.pre_quant import compute_modality_gradients
except ModuleNotFoundError:
    compute_modality_gradients = None  # MBQ not present in this repo
try:
    from qvla.methods.ostquant.entry import patch_action_head_unrotate_r_res
except ModuleNotFoundError:
    patch_action_head_unrotate_r_res = None  # OSTQuant not present in this repo
from libero.libero import benchmark

def _set_reproducibility(seed: int) -> None:
    import random as _py_random

    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_generate_config(args) -> GenerateConfig:
    return GenerateConfig(
        pretrained_checkpoint=args.model,
        use_l1_regression=True,
        use_diffusion=False,
        num_images_in_input=args.num_images,
        use_proprio=args.use_proprio,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=args.unnorm_key if args.unnorm_key else "",
        load_in_8bit=False,
        load_in_4bit=False,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
    )


def parse_quant_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default="", help="YAML config; when set, CLI args are ignored.")

    def _parse_bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return True
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean value, got {v!r}")

    # Model / data
    parser.add_argument("--model", required=True, help="OpenVLA checkpoint path or HF repo id.")
    parser.add_argument(
        "--data",
        required=True,
        help="TFDS version dir for LIBERO RLDS data (must contain dataset_info.json), e.g. .../libero_spatial_no_noops/1.0.0",
    )
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument(
        "--rotate_images",
        action="store_true",
        help="Rotate all images 180° (not part of official preprocessing; keep disabled for strict train-time alignment).",
    )
    parser.add_argument("--unnorm_key", type=str, default=None, help="Choose norm_stats key (default: first one).")
    parser.add_argument("--num_images", type=int, default=2, help="#images per sample (primary + wrist).")
    parser.add_argument("--use_proprio", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true", help="Disable dataset shuffling.")
    parser.add_argument(
        "--calib_sampling",
        default="random_single",
        choices=["random_single", "proprio_stratified"],
        help="Calibration sampling strategy (random one-timestep or proprio+image stratified).",
    )
    parser.add_argument(
        "--calib_prompt_balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ensure calibration covers all language prompts when using stratified sampling.",
    )
    parser.add_argument(
        "--calib_print_selection",
        action="store_true",
        help="Print selected calibration items (instruction + traj/episode index + timestep).",
    )
    parser.add_argument(
        "--calib_print_samples",
        type=int,
        default=50,
        help="Max number of selected calibration samples to print when --calib_print_selection is set (<=0 prints all).",
    )

    # Quantization
    # --- modified by lyy 12.18 ---
    parser.add_argument(
        "--method",
        default="mbq",
        choices=["awq", "smoothquant", "mbq", "rtn", "spinquant", "omniquant", "ostquant"],
    )
    # --- end modified by lyy 12.18 ---
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument(
        "--weight_quant",
        default="auto",
        choices=["auto", "per_group", "per_channel", "per_tensor"],
        help="Weight quantization granularity for RTN (auto keeps old defaults).",
    )
    parser.add_argument(
        "--act_quant",
        default="per_token",
        choices=["per_token", "per_tensor"],
        help="Activation quantization granularity for RTN weight+activation quantization.",
    )
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--sq_scale_mode",
        default="max",
        choices=["max", "weighted_max", "quantile"],
        help="SmoothQuant activation scale statistic (max matches official SmoothQuant).",
    )
    parser.add_argument(
        "--sq_target",
        default="llm",
        choices=["llm", "vision"],
        help="SmoothQuant target module (`llm` keeps legacy behavior; `vision` quantizes timm vision encoder only).",
    )
    parser.add_argument(
        "--sq_quantile",
        type=float,
        default=0.999,
        help="Quantile used when --sq_scale_mode=quantile (per-channel).",
    )
    parser.add_argument("--sq_w_vision", type=float, default=1.0, help="Token weight for vision_patch_mask (weighted_max).")
    parser.add_argument("--sq_w_proprio", type=float, default=1.0, help="Token weight for proprio_mask (weighted_max).")
    parser.add_argument("--sq_w_text", type=float, default=1.0, help="Token weight for text_mask (weighted_max).")
    parser.add_argument("--sq_w_action", type=float, default=1.0, help="Token weight for action_mask (weighted_max).")
    parser.add_argument(
        "--sq_w_asym",
        action="store_true",
        help="Use asymmetric weight quantization (zero-point) for SmoothQuant.",
    )
    parser.add_argument(
        "--sq_a_asym",
        action="store_true",
        help="Use asymmetric activation quantization (zero-point) for SmoothQuant.",
    )
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--wa_quant", action="store_true", help="Enable weight-activation quantization (MBQ)")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    # OmniQuant
    parser.add_argument("--omni_epochs", type=int, default=10, help="OmniQuant training epochs (LET/LWC).")
    parser.add_argument("--omni_batch_size", type=int, default=1, help="OmniQuant batch size for calibration.")
    parser.add_argument("--omni_let", action="store_true", help="Enable OmniQuant LET (learnable smoothing).")
    parser.add_argument("--omni_lwc", action="store_true", help="Enable OmniQuant LWC (learnable weight clipping).")
    parser.add_argument("--omni_let_lr", type=float, default=5e-3, help="OmniQuant LET learning rate.")
    parser.add_argument("--omni_lwc_lr", type=float, default=1e-2, help="OmniQuant LWC learning rate.")
    parser.add_argument("--omni_wd", type=float, default=0.0, help="OmniQuant weight decay.")
    parser.add_argument("--omni_aug_loss", action="store_true", help="Enable OmniQuant augmented loss.")
    parser.add_argument("--omni_symmetric", action="store_true", help="Use symmetric quantization in OmniQuant.")
    parser.add_argument("--omni_disable_zero_point", action="store_true", help="Disable zero-point in OmniQuant.")
    parser.add_argument(
        "--omni_a_dynamic_method",
        type=str,
        default="per_token",
        choices=["per_token"],
        help="OmniQuant activation dynamic method.",
    )
    parser.add_argument(
        "--omni_w_dynamic_method",
        type=str,
        default="per_channel",
        choices=["per_channel"],
        help="OmniQuant weight dynamic method.",
    )
    parser.add_argument("--omni_deactive_amp", action="store_true", help="Disable AMP inside OmniQuant.")
    parser.add_argument("--omni_real_quant", action="store_true", help="Enable OmniQuant real quantization (GPTQ).")
    # --- modified by lyy 12.18 ---
    parser.add_argument(
        "--spin_rotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SpinQuant structured rotations (default: True when method=spinquant).",
    )
    # spinquantv2: opt-in patch mode (keeps original spinquant unchanged when disabled).
    parser.add_argument(
        "--spin_quantv2",
        action="store_true",
        help="Enable SpinQuantV2: add an auxiliary R2 quant-reconstruction loss during rotation learning.",
    )
    parser.add_argument(
        "--spin_v2_r2_mse_weight",
        type=float,
        default=1.0,
        help="SpinQuantV2: weight for the R2 quant-reconstruction MSE loss (added to action_loss).",
    )
    # xiugai_lyy: allow choosing recon loss type (MSE vs MAE) for SpinQuantV2
    parser.add_argument(
        "--spin_v2_recon_loss_mode",
        type=str,
        default="mse",
        choices=["mse", "mae"],
        help="SpinQuantV2: recon loss type for auxiliary R2 reconstruction (mse|mae).",
    )
    parser.add_argument(
        "--spin_v2_modality_weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SpinQuantV2: use MBQ-style modality gradient stats to weight the R2 MSE loss (default: True).",
    )
    parser.add_argument(
        "--spin_v2_weight_keys",
        type=str,
        default="",
        help=(
            "SpinQuantV2: comma-separated module-name substrings to use when computing modality weights "
            "(e.g., 'o_proj,wo' or 'v_proj'). Empty keeps the default behavior."
        ),
    )
    parser.add_argument(
        "--spin_v2_grad_dump_path",
        type=str,
        default=None,
        help="SpinQuantV2: optional path to save detailed modality grad/token stats (txt).",
    )
    parser.add_argument(
        "--spin_v2_grad_hook_target",
        type=str,
        default="linear",
        choices=["linear", "block"],
        help="SpinQuantV2: where to attach backward hooks for grad stats (default: linear).",
    )
    parser.add_argument(
        "--spin_v2_grad_hook_scope",
        type=str,
        default="proj_linear",
        choices=["proj_linear", "all_linear", "all_modules"],
        help="SpinQuantV2: which modules to hook when hook_target includes 'linear' (default: proj_linear).",
    )
    parser.add_argument(
        "--spin_lr",
        type=float,
        default=1e-2,
        help="Learning rate for SpinQuant rotation optimization (`--method spinquant --run_process`).",
    )
    parser.add_argument(
        "--spin_steps",
        type=int,
        default=200,
        help="Optimization steps for SpinQuant rotation learning.",
    )
    parser.add_argument(
        "--spin_batch_size",
        type=int,
        default=1,
        help="Batch size for SpinQuant rotation learning (uses calibration samples).",
    )
    parser.add_argument(
        "--spin_log_every",
        type=int,
        default=10,
        help="Log frequency (steps) for SpinQuant rotation learning.",
    )
    parser.add_argument(
        "--spin_overwrite",
        action="store_true",
        help="Overwrite existing `--scale_path` when running SpinQuant rotation learning.",
    )
    parser.add_argument(
        "--spin_fp32_had",
        action="store_true",
        help="Compute SpinQuant Hadamard rotations in FP32 for stability (slower).",
    )
    parser.add_argument(
        "--spin_w_asym",
        action="store_true",
        help="Use asymmetric weight quantization (zero-point) for SpinQuant.",
    )
    parser.add_argument(
        "--spin_w_rtn",
        action="store_true",
        help="Use RTN for SpinQuant weight PTQ (default is GPTQ when w_bit < 16, matching official SpinQuant).",
    )
    parser.add_argument(
        "--spin_gptq_percdamp",
        type=float,
        default=0.01,
        help="GPTQ percdamp for SpinQuant weight PTQ (only used when --spin_w_rtn is not set).",
    )
    parser.add_argument(
        "--spin_gptq_blocksize",
        type=int,
        default=128,
        help="GPTQ blocksize for SpinQuant weight PTQ (only used when --spin_w_rtn is not set).",
    )
    parser.add_argument(
        "--spin_gptq_actorder",
        action="store_true",
        help="Enable GPTQ act-order for SpinQuant weight PTQ (only used when --spin_w_rtn is not set).",
    )
    parser.add_argument(
        "--spin_a_asym",
        action="store_true",
        help="Use asymmetric activation quantization (zero-point) for SpinQuant.",
    )
    parser.add_argument(
        "--spin_int8_down_proj",
        action="store_true",
        help="Force INT8 quantization for MLP down_proj weights (SpinQuant behavior).",
    )
    parser.add_argument(
        "--spin_grad_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing during SpinQuant rotation learning to reduce VRAM.",
    )
    parser.add_argument(
        "--spin_w_clip",
        action="store_true",
        help="Enable SpinQuant weight clipping (MSE grid search) during fake-quant weight param selection.",
    )
    parser.add_argument("--spin_a_clip_ratio", type=float, default=1.0, help="Activation clip ratio for SpinQuant.")
    parser.add_argument("--spin_k_clip_ratio", type=float, default=1.0, help="K-cache clip ratio for SpinQuant.")
    parser.add_argument("--spin_v_clip_ratio", type=float, default=1.0, help="V-cache clip ratio for SpinQuant.")
    parser.add_argument("--k_bit", type=int, default=16, help="K-cache quantization bits for SpinQuant (default: 16).")
    parser.add_argument("--v_bit", type=int, default=16, help="V-cache quantization bits for SpinQuant (default: 16).")
    parser.add_argument(
        "--k_group",
        type=int,
        default=-1,
        help="K-cache groupsize for SpinQuant (-1 token-wise; or head_dim for head-wise).",
    )
    parser.add_argument("--v_group", type=int, default=-1, help="V-cache groupsize for SpinQuant (kept for parity).")
    parser.add_argument("--spin_k_asym", action="store_true", help="Use asymmetric K-cache quantization for SpinQuant.")
    parser.add_argument("--spin_v_asym", action="store_true", help="Use asymmetric V-cache quantization for SpinQuant.")
    parser.add_argument(
        "--spin_k_pre_rope",
        action="store_true",
        help="Pre-RoPE K quantization (official SpinQuant: not supported).",
    )
    # --- end modified by lyy 12.18 ---
    # (DFRot args removed)
    # --- modified by lyy ---
    parser.add_argument(
        "--capture_modality_grad",
        action="store_true",
        help="During MBQ calibration, run forward+backward and save per-modality gradient magnitudes "
        "(vision/proprio/text/action) into the MBQ results file at `--scale_path`.",
    )
    parser.add_argument(
        "--analyze_modality_grad",
        action="store_true",
        help="Analyze modality gradient importance on the full-precision model (no quantization). "
        "Writes a txt summary and exits; does not require `--scale_path`.",
    )
    parser.add_argument(
        "--grad_out",
        type=str,
        default="",
        help="Output txt path for `--analyze_modality_grad` (default: <model_name>_modality_grad_fp.txt).",
    )
    # --- end modified by lyy ---
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument(
        "--sq_vision_scale_path",
        default=None,
        type=str,
        help="Path to save/load SmoothQuant vision scales (required when --method smoothquant --sq_target vision).",
    )
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    # evaluation switch
    parser.add_argument("--eval_libero", action="store_true", help="Run LIBERO evaluation after quantization")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)

    # === OSTQuant (VLA LM-only) ===
    parser.add_argument("--pre_eval", action="store_true", help="OSTQuant: run float eval before quantization (rank0).")
    parser.add_argument("--output_dir", type=str, default="", help="OSTQuant: output dir; saves `model.bin` here.")
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OSTQuant: enable rotation/smoothing stage (keep True for OSTQuant runs).",
    )
    parser.add_argument("--train_rotate", action="store_true", help="OSTQuant: train rotation/smoothing parameters.")
    parser.add_argument("--resume_path", type=str, default=None, help="OSTQuant: load trained params (model.bin).")
    parser.add_argument(
        "--train_distribute",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "OSTQuant: single-process multi-GPU model dispatch for training (matches official ostq `--train_distribute`). "
            "Use with `CUDA_VISIBLE_DEVICES=... python main.py` (do NOT combine with torchrun)."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "OSTQuant: enable gradient checkpointing during rotate/smooth training. "
            "Note: OSTQuant training uses `inputs_embeds`; when enabled, we force `inputs_embeds.requires_grad=True` "
            "to keep checkpointing compatible."
        ),
    )

    # --- modified by lyy ---
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl_top",
        help="OSTQuant: training loss (kl_top/kl/origin/action_hidden/action_pred_l1).",
    )
    # --- end modified by lyy ---
    parser.add_argument(
        "--post_attn",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "OSTQuant: for kl_top loss, compute KL on post-softmax full-vocab probabilities then gather top-k "
            "(matches official ostq). Accepts `--post_attn`, `--post_attn=true/false`."
        ),
    )
    parser.add_argument("--opt_type", type=str, default="RAdam", choices=["SGDG", "RAdam", "RSGD"], help="OSTQuant: optimizer.")
    parser.add_argument("--rotate_lr", type=float, default=1e-2, help="OSTQuant: rotation lr.")
    parser.add_argument("--smooth_lr", type=float, default=1e-3, help="OSTQuant: smoothing lr.")
    parser.add_argument("--rotate_momentom", type=float, default=0.0)
    parser.add_argument("--smooth_momentom", type=float, default=0.9)
    parser.add_argument("--train_enable_wquant", action="store_true", help="OSTQuant: enable weight quant during training.")
    parser.add_argument("--fully_quant", action="store_true", help="OSTQuant: enable fully-quant ops.")
    parser.add_argument("--bf16", action="store_true", help="OSTQuant: train in bf16.")
    parser.add_argument("--fp16", action="store_true", help="OSTQuant: train in fp16.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="OSTQuant: gradient accumulation steps (effective batch = per_device_train_batch_size * steps).",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0)

    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--k_bits", type=int, default=4)
    parser.add_argument("--v_bits", type=int, default=4)
    parser.add_argument("--down_bits", type=int, default=4)
    parser.add_argument("--residual_bits", type=int, default=16)
    parser.add_argument("--act_bits", type=int, default=16)
    parser.add_argument("--attn_bits", type=int, default=16)

    parser.add_argument("--w_groupsize", type=int, default=-1)
    parser.add_argument("--a_groupsize", type=int, default=-1)
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument("--a_dynamic_method", type=str, default="pertoken", choices=["pertoken", "perchannel", "pertensor"])
    parser.add_argument("--a_asym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--w_asym", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--k_asym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--v_asym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--a_clip_ratio", type=float, default=1.0)
    parser.add_argument("--k_clip_ratio", type=float, default=1.0)
    parser.add_argument("--v_clip_ratio", type=float, default=1.0)
    parser.add_argument("--w_clip", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--w_gptq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--act_order", action="store_true")

    parser.add_argument("--sub_mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use_klt",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "OSTQuant: initialize R_res (and R_ov when enabled) with KLT (PCA) as in official ostq. "
            "Accepts `--use_klt`, `--use_klt=true/false`."
        ),
    )
    parser.add_argument("--rotate_ov", action="store_true", help="OSTQuant: rotate V's output and O's input.")
    parser.add_argument("--rotate_pre_rope", action="store_true", help="OSTQuant: rotate ROPE's input.")
    parser.add_argument("--rotate_post_rope", action="store_true", help="OSTQuant: rotate ROPE's output.")
    parser.add_argument(
        "--rotate_rope_perlayer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OSTQuant: whether to allow each layer to have a separate ROPE matrix.",
    )
    parser.add_argument("--online_hadamard", type=str, default="down", choices=["all", "v", "down", "None"])
    parser.add_argument("--online_qk_hadamard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32_had", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smooth_up_down", action="store_true")
    parser.add_argument("--smooth_up_gate", action="store_true")
    parser.add_argument("--smooth_qk", action="store_true")
    parser.add_argument("--smooth_ov", action="store_true")
    parser.add_argument("--smooth_norm_linear", action="store_true")
    parser.add_argument("--rotate_down_dim", type=int, default=1)
    return parser.parse_args()


def _print_calib_selection(args: argparse.Namespace, debug: dict) -> None:
    instrs = list(debug.get("instruction", []))
    traj_ids = list(debug.get("traj_index", []))
    timesteps = list(debug.get("timestep", []))
    ds_names = list(debug.get("dataset_name", []))

    if not instrs:
        print("\n[calibration selection] No debug metadata returned.", flush=True)
        return

    h = hashlib.sha256()
    for traj, t, instr in zip(traj_ids, timesteps, instrs):
        h.update(f"{traj}\t{t}\t{instr}\n".encode("utf-8"))
    fingerprint = h.hexdigest()[:16]

    print("\n[calibration selection settings]", flush=True)
    print(
        f"n_samples={int(args.n_samples)}  seed={int(args.seed)}  shuffle={not bool(args.deterministic)}  "
        f"calib_sampling={args.calib_sampling}  calib_prompt_balance={bool(args.calib_prompt_balance)}",
        flush=True,
    )
    print(f"selection_fingerprint_sha256[:16]={fingerprint}", flush=True)

    counts: dict[str, int] = {}
    for s in instrs:
        s = str(s)
        counts[s] = counts.get(s, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print("\n[selected prompts]", flush=True)
    for s, c in top[: min(20, len(top))]:
        print(f"{c:4d}  {s}", flush=True)
    if len(top) > 20:
        print(f"... ({len(top) - 20} more prompts)", flush=True)

    n = min(int(args.calib_print_samples), len(instrs)) if int(args.calib_print_samples) > 0 else len(instrs)
    print("\n[selected samples] (traj_index, timestep -> instruction)", flush=True)
    for i in range(n):
        ds = ds_names[i] if i < len(ds_names) else ""
        print(f"{i:4d}  traj={traj_ids[i]}  t={timesteps[i]}  ds={ds}  prompt={instrs[i]}", flush=True)


def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        args = parse_quant_args()

    args_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if not isinstance(config_args, list) else config_args
        for cfg_dict in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in cfg_dict.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    for cfg_args in args_list:
        cli_quant_single(cfg_args)


def cli_quant_single(args: argparse.Namespace) -> None:
    # torchrun/DDP: ensure each rank uses its own local GPU (CUDA_VISIBLE_DEVICES remaps to cuda:0..).
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Reproducibility: calibration sampling (python/tf) + gradient analysis/backward (torch/numpy).
    _set_reproducibility(args.seed)

    # Build minimal GenerateConfig mirroring inference defaults
    cfg = _build_generate_config(args)

    # --- Model + processors ---
    vla = get_vla(cfg)
    # Resolve unnorm_key if not explicitly provided
    check_unnorm_key(cfg, vla)
    args.unnorm_key = cfg.unnorm_key
    processor = get_processor(cfg)
    proprio_projector = (
        get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM) if args.use_proprio else None
    )
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    # Ensure deterministic behavior (no dropout) for gradient importance runs.
    vla.eval()
    if proprio_projector is not None:
        proprio_projector.eval()
    if action_head is not None:
        action_head.eval()

    # --- Optional pre-eval (float, before quantization) ---
    if args.method == "ostquant" and bool(getattr(args, "pre_eval", False)) and args.eval_libero:
        import torch.distributed as dist

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            validate_config(cfg)
            set_seed_everywhere(cfg.seed)
            resize_size = get_image_resize_size(cfg)
            noisy_action_projector = None
            if cfg.use_diffusion:
                noisy_action_projector = get_noisy_action_projector(cfg, vla.llm_dim)
            log_file, _, _ = setup_logging(cfg)
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[cfg.task_suite_name]()
            num_tasks = task_suite.n_tasks
            total_episodes, total_successes = 0, 0
            for task_id in tqdm.tqdm(range(num_tasks)):
                total_episodes, total_successes = run_task(
                    cfg,
                    task_suite,
                    task_id,
                    vla,
                    resize_size,
                    processor,
                    action_head,
                    proprio_projector,
                    noisy_action_projector,
                    total_episodes,
                    total_successes,
                    log_file,
                )
            final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
            print(
                f"[Float Eval] Total episodes: {total_episodes}, successes: {total_successes}, "
                f"success_rate: {final_success_rate:.4f} ({final_success_rate*100:.1f}%)"
            )
            if log_file:
                log_file.close()

    # --- Calibration inputs ---
    calib_return_mode = "embeds"
    if args.method == "smoothquant" and args.sq_target == "vision":
        calib_return_mode = "raw"

    calib_out = get_libero_calib_dataset(
        data_path=args.data,
        model=vla,
        processor=processor,
        proprio_projector=proprio_projector,
        n_samples=args.n_samples,
        rotate=args.rotate_images,
        shuffle=not args.deterministic,
        unnorm_key=args.unnorm_key,
        # --- modified by lyy ---
        seed=args.seed,
        calib_sampling=args.calib_sampling,
        calib_prompt_balance=args.calib_prompt_balance,
        # --- end modified by lyy ---
        return_mode=calib_return_mode,
        return_debug=bool(args.calib_print_selection),
    )

    if bool(args.calib_print_selection):
        prompt_inputs, prompt_kwargs, debug = calib_out
        _print_calib_selection(args, debug)
    else:
        prompt_inputs, prompt_kwargs = calib_out

    # --- Quantization ---
    # qwrapper returns the (possibly) quantized model; capture it to ensure eval uses the quantized version.
    vla = qwrapper(
        vla,
        prompt_inputs,
        prompt_kwargs,
        args,
        action_head=action_head,
        grad_loss_type="l1_regression",
    )
    if args.method == "ostquant":
        patch_action_head_unrotate_r_res(action_head, vla_model=vla)
    # Optional LIBERO evaluation with the quantized model
    if args.eval_libero:
        import torch.distributed as dist

        # For multi-process OSTQuant training under torchrun, only run final eval on rank0.
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        # The model is already quantized in-place. Just run eval using the official loop.
        validate_config(cfg)
        set_seed_everywhere(cfg.seed)
        resize_size = get_image_resize_size(cfg)

        # Reuse action_head / proprio_projector / processor / noisy_action_projector
        noisy_action_projector = None
        if cfg.use_diffusion:
            noisy_action_projector = get_noisy_action_projector(cfg, vla.llm_dim)

        # Setup logging
        log_file, _, _ = setup_logging(cfg)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks = task_suite.n_tasks

        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks)):
            total_episodes, total_successes = run_task(
                cfg,
                task_suite,
                task_id,
                vla,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
            )

        final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
        print(f"[Quantized Eval] Total episodes: {total_episodes}, successes: {total_successes}, "
              f"success_rate: {final_success_rate:.4f} ({final_success_rate*100:.1f}%)")

        if log_file:
            log_file.close()


if __name__ == "__main__":
    cli_quant()
