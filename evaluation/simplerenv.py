"""Reuse the migrated policy adapters; no simulator or model code is duplicated here."""
from pathlib import Path
import os
import sys


def migrated_command(args, quant_root, env):
    name = "SimplerEnv-OpenVLA" if args.model_backend == "openvla" else "SimplerEnv-OpenVLA-OFT"
    root = Path(args.simplerenv_root or quant_root.parent / "zwt" / name).expanduser().resolve()
    entry = root / "main_quant.py"
    if not entry.is_file():
        raise FileNotFoundError(f"Migrated SimplerEnv entry missing: {entry}; set --simplerenv_root.")
    mode = f"simplerenv_{args.mode}"
    if args.benchmark_subset:
        if args.mode != "fp" or args.model_backend != "openvla":
            raise ValueError("Official subsets are currently wired only for OpenVLA FP.")
        mode = "simplerenv_official"
    elif not args.tasks_json:
        raise ValueError("SimplerEnv requires --tasks_json or an OpenVLA FP --benchmark_subset.")
    model = Path(args.checkpoint).expanduser()
    command = [sys.executable, str(entry), "--eval_mode", mode, "--model_backend", args.model_backend,
               "--model", str(model.resolve()) if model.exists() else args.checkpoint,
               "--logging_dir", str(Path(args.output_dir).expanduser().resolve() / "rollouts")]
    if args.tasks_json:
        command += ["--simplerenv_tasks_json", str(Path(args.tasks_json).expanduser().resolve())]
    if args.benchmark_subset:
        command += ["--benchmark_subset", args.benchmark_subset]
    if args.model_backend == "openvla_oft":
        command += ["--use_l1_regression", "--num_images", str(args.num_images)]
        if args.use_proprio:
            command.append("--use_proprio")
    if args.unnorm_key:
        command += ["--unnorm_key", args.unnorm_key]
    command += ["--seed", str(args.seed)]
    if args.mode == "quant":
        # The migrated quantizer does not expose explicit symmetry switches yet.
        # Do not silently accept a precision request that that adapter cannot enforce.
        raise NotImplementedError("Migrated SimplerEnv quantization still needs symmetry/config validation; "
                                  "its source is preserved, but this unified route is not enabled yet.")
    env = dict(env)
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "ManiSkill2_real2sim")])
    return command, root, env
