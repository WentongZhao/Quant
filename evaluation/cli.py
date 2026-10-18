"""Parse experiment configuration without importing GPU or simulator libraries."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def build_parser():
    p = argparse.ArgumentParser(description="Unified VLA evaluation (FP and quantized policies).")
    p.add_argument("--config", help="YAML mapping; explicitly supplied CLI values override it.")
    p.add_argument("--list", action="store_true", help="List wired backend/benchmark/mode combinations.")
    p.add_argument("--dry_run", action="store_true", help="Validate configuration and print the backend job without running it.")
    # Suppress defaults so CLI values can be distinguished from configuration defaults.
    p.add_argument("--model_backend", default=argparse.SUPPRESS)
    p.add_argument("--benchmark", choices=["libero", "simplerenv"], default=argparse.SUPPRESS)
    p.add_argument("--mode", choices=["fp", "quant"], default=argparse.SUPPRESS)
    p.add_argument("--checkpoint", default=argparse.SUPPRESS)
    p.add_argument("--output_dir", default=argparse.SUPPRESS)
    p.add_argument("--data", default=argparse.SUPPRESS, help="Calibration dataset; unnecessary for FP.")
    p.add_argument("--task_suite_name", default=argparse.SUPPRESS)
    p.add_argument("--num_trials_per_task", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval_task_ids", type=int, nargs="+", default=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    p.add_argument("--num_images", type=int, default=argparse.SUPPRESS)
    p.add_argument("--use_proprio", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--method", default=argparse.SUPPRESS)
    for flag in ("w_bit", "a_bit", "k_bit", "v_bit", "n_samples"):
        p.add_argument("--" + flag, type=int, default=argparse.SUPPRESS)
    p.add_argument("--alpha", type=float, default=argparse.SUPPRESS)
    p.add_argument("--sq_target", choices=["llm", "vision"], default=argparse.SUPPRESS)
    p.add_argument("--sq_scale_mode", choices=["max", "weighted_max", "quantile"], default=argparse.SUPPRESS)
    p.add_argument("--scale_path", default=argparse.SUPPRESS)
    p.add_argument("--sq_vision_scale_path", default=argparse.SUPPRESS)
    p.add_argument("--calib_sampling", choices=["random_single", "proprio_stratified"], default=argparse.SUPPRESS)
    for flag in ("run_process", "sq_w_asym", "sq_a_asym", "deterministic", "calib_prompt_balance"):
        p.add_argument("--" + flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    p.add_argument("--unnorm_key", default=argparse.SUPPRESS)
    p.add_argument("--simplerenv_root", default=argparse.SUPPRESS, help="Migrated SimplerEnv checkout to call.")
    p.add_argument("--tasks_json", default=argparse.SUPPRESS)
    p.add_argument("--benchmark_subset", default=argparse.SUPPRESS)
    return p


DEFAULTS = dict(
    model_backend="openvla_oft", benchmark="libero", mode="fp", checkpoint=None,
    output_dir=None, data="", task_suite_name="libero_spatial", num_trials_per_task=50,
    eval_task_ids=None, seed=7, num_images=2, use_proprio=True, method="smoothquant",
    w_bit=8, a_bit=8, k_bit=16, v_bit=16, n_samples=256, alpha=0.5, sq_target="llm",
    sq_scale_mode="max", scale_path=None, sq_vision_scale_path=None,
    calib_sampling="proprio_stratified", run_process=False, sq_w_asym=False,
    sq_a_asym=True, deterministic=True, calib_prompt_balance=True, unnorm_key=None,
    simplerenv_root=None, tasks_json=None, benchmark_subset=None,
)


def parse_args(argv=None):
    parser = build_parser()
    cli = vars(parser.parse_args(argv))
    config = {}
    if cli.get("config"):
        import yaml

        path = Path(cli["config"]).expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            parser.error("--config must contain a single YAML mapping.")
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            parser.error("Unknown config keys: " + ", ".join(sorted(unknown)))
        # Relative paths in YAML belong to the YAML file, not the launch directory.
        for key in ("output_dir", "data", "scale_path", "sq_vision_scale_path", "simplerenv_root", "tasks_json"):
            if config.get(key) and not Path(config[key]).is_absolute():
                config[key] = str((path.parent / config[key]).resolve())
        checkpoint = config.get("checkpoint")
        if checkpoint and checkpoint.startswith(("./", "../")):
            config["checkpoint"] = str((path.parent / checkpoint).resolve())
    values = {**DEFAULTS, **config, **cli}
    if values["list"]:
        return argparse.Namespace(**values)
    if not values["checkpoint"]:
        parser.error("--checkpoint is required (CLI or YAML).")
    if values["mode"] not in {"fp", "quant"} or values["benchmark"] not in {"libero", "simplerenv"}:
        parser.error("Invalid benchmark or mode in configuration.")
    for key in ("run_process", "sq_w_asym", "sq_a_asym", "deterministic", "calib_prompt_balance", "use_proprio"):
        if not isinstance(values[key], bool):
            parser.error(f"{key} must be a YAML boolean (true/false), not a quoted string.")
    for key in ("num_trials_per_task", "num_images", "n_samples", "w_bit", "a_bit", "k_bit", "v_bit", "seed"):
        if type(values[key]) is not int:
            parser.error(f"{key} must be an integer.")
    for key, choices in {
        "sq_target": {"llm", "vision"}, "sq_scale_mode": {"max", "weighted_max", "quantile"},
        "calib_sampling": {"random_single", "proprio_stratified"},
    }.items():
        if values[key] not in choices:
            parser.error(f"Invalid {key}: {values[key]}")
    if values["num_trials_per_task"] <= 0:
        parser.error("--num_trials_per_task must be positive.")
    if values["n_samples"] <= 0 or values["num_images"] <= 0:
        parser.error("--n_samples and --num_images must be positive.")
    if values["eval_task_ids"] is not None:
        ids = values["eval_task_ids"]
        if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids) or len(set(ids)) != len(ids):
            parser.error("--eval_task_ids must contain unique nonnegative integers.")
    if values["mode"] == "fp" and values["run_process"]:
        parser.error("FP evaluation cannot request quantization calibration (--run_process).")
    if values["mode"] == "quant":
        if not values["data"]:
            parser.error("Quantized evaluation requires --data for calibration inputs.")
        if values["method"] != "smoothquant":
            parser.error("The unified quantized runner currently validates SmoothQuant only; other methods remain in main_quant.py.")
        if any(not 2 <= values[k] <= 16 for k in ("w_bit", "a_bit", "k_bit", "v_bit")):
            parser.error("Bit widths must be in [2, 16].")
        if values["k_bit"] != 16 or values["v_bit"] != 16:
            parser.error("SmoothQuant does not quantize the KV cache in this runner; use --k_bit 16 --v_bit 16.")
        if not isinstance(values["alpha"], (int, float)) or not 0 <= values["alpha"] <= 1:
            parser.error("SmoothQuant alpha must be in [0, 1].")
        scale_key = "scale_path" if values["sq_target"] == "llm" else "sq_vision_scale_path"
        if not values[scale_key]:
            parser.error("SmoothQuant requires --" + scale_key + ".")
        if not values["run_process"] and not Path(values[scale_key]).is_file():
            parser.error("Scale file is missing; supply an existing file or --run_process.")
    if values["output_dir"] is None:
        tag = f"{values['model_backend']}-{values['benchmark']}-{values['mode']}"
        if values["mode"] == "quant":
            tag += f"-w{values['w_bit']}a{values['a_bit']}"
        values["output_dir"] = str(Path(__file__).resolve().parents[1] / "runs" /
                                   (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + tag))
    return argparse.Namespace(**values)


def main(argv=None):
    from evaluation.dispatch import run_evaluation, describe_runners

    args = parse_args(argv)
    if args.list:
        print(describe_runners())
        return
    run_evaluation(args)
