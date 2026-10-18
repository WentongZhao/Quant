"""Dispatch supported combinations while keeping backend import environments isolated."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


RUNNERS = {
    ("openvla_oft", "libero", "fp"): "libero",
    ("openvla_oft", "libero", "quant"): "libero",
    ("openvla", "simplerenv", "fp"): "simplerenv",
    ("openvla_oft", "simplerenv", "fp"): "simplerenv",
}


def describe_runners():
    return "\n".join(
        f"{model:12} {benchmark:10} {mode:5}  " +
        ("shared LIBERO worker" if runner == "libero" else "migrated adapter; requires separate setup/validation")
        for (model, benchmark, mode), runner in RUNNERS.items()
    )


def libero_config(args):
    fields = (
        "model_backend", "data", "task_suite_name", "num_trials_per_task", "eval_task_ids", "seed",
        "num_images", "use_proprio", "method", "w_bit", "a_bit", "k_bit", "v_bit", "n_samples",
        "alpha", "sq_target", "sq_scale_mode", "scale_path", "sq_vision_scale_path", "calib_sampling",
        "run_process", "sq_w_asym", "sq_a_asym", "deterministic", "calib_prompt_balance", "unnorm_key",
    )
    config = {name: getattr(args, name) for name in fields}
    model_path = Path(args.checkpoint).expanduser()
    output = Path(args.output_dir).expanduser().resolve()
    config.update(
        model=str(model_path.resolve()) if model_path.exists() else args.checkpoint,
        eval_mode=f"libero_{args.mode}", eval_libero=True, pseudo_quant=args.mode == "quant",
        use_l1_regression=True, results_json=str(output / "results.json"),
        local_log_dir=str(output / "logs"), run_id_note=output.name,
    )
    for name in ("data", "scale_path", "sq_vision_scale_path"):
        if config[name]:
            config[name] = str(Path(config[name]).expanduser().resolve())
    return config


def run_evaluation(args):
    key = (args.model_backend, args.benchmark, args.mode)
    if key not in RUNNERS:
        raise ValueError(f"Combination not wired: {key}. Run evaluate.py --list for available routes.")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if RUNNERS[key] == "libero":
        config = libero_config(args)
        config_path = output / "backend_config.json"
        command = [sys.executable, str(root / "main_quant.py"), "--config", str(config_path)]
        # Native rollout paths are relative to cwd: isolate every run's videos.
        cwd = output
        env["PYTHONPATH"] = str(root)
    else:
        from evaluation.simplerenv import migrated_command

        command, cwd, env = migrated_command(args, root, env)
        config = None
    if args.dry_run:
        print(json.dumps({"command": command, "cwd": str(cwd), "backend_config": config}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists() or (output / "results.json").exists():
        raise FileExistsError(f"Output directory already contains a run: {output}; select a new --output_dir.")
    if config is not None:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    record = {"status": "running", "request": vars(args), "command": command, "cwd": str(cwd)}
    manifest = output / "run.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[evaluate] {key} -> {command[1]}\n[evaluate] output={output}", flush=True)
    try:
        result = subprocess.run(command, cwd=str(cwd), env=env, check=False)
        record.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
