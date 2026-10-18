"""LIBERO task loop shared by floating-point and quantized policies."""
import json
import os
import time
from pathlib import Path

import torch
import tqdm
from libero.libero import benchmark
from experiments.robot.libero.run_libero_eval import validate_config, setup_logging, run_task
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere


def evaluate_libero(args, cfg, vla, processor, action_head, proprio_projector, *, quantized):
    """Use exactly the same task loop and model components for FP and quantized policies."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    resize_size = get_image_resize_size(cfg)
    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task_ids = list(range(task_suite.n_tasks)) if args.eval_task_ids is None else list(args.eval_task_ids)
    if not task_ids or len(task_ids) != len(set(task_ids)) or any(t < 0 or t >= task_suite.n_tasks for t in task_ids):
        raise ValueError(f"--eval_task_ids must be unique IDs in [0, {task_suite.n_tasks}).")
    log_file, log_path, run_id = setup_logging(cfg)
    result_path = Path(args.results_json) if args.results_json else Path(log_path).with_suffix(".json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1, "status": "running", "model_backend": "openvla_oft",
        "benchmark": "libero", "eval_mode": "libero_quant" if quantized else "libero_fp",
        "model": str(args.model), "task_suite": cfg.task_suite_name, "task_ids": task_ids,
        "trials_per_task": cfg.num_trials_per_task, "seed": cfg.seed,
        "unnorm_key": cfg.unnorm_key, "num_images": cfg.num_images_in_input,
        "use_proprio": cfg.use_proprio, "num_open_loop_steps": cfg.num_open_loop_steps,
        "dtype": str(next(vla.parameters()).dtype), "run_id": run_id, "log_path": log_path,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "attention_implementation": getattr(vla.language_model.config, "_attn_implementation", None),
        "quantization": None if not quantized else {
            "method": args.method, "w_bit": args.w_bit, "a_bit": args.a_bit,
            "k_bit": args.k_bit, "v_bit": args.v_bit, "pseudo_quant": args.pseudo_quant,
            "sq_target": args.sq_target, "sq_w_asym": args.sq_w_asym, "sq_a_asym": args.sq_a_asym,
            "alpha": args.alpha, "scale_path": args.scale_path,
            "sq_vision_scale_path": args.sq_vision_scale_path, "sq_scale_mode": args.sq_scale_mode,
            "n_samples": args.n_samples, "run_process": args.run_process,
            "calib_sampling": args.calib_sampling, "calib_prompt_balance": args.calib_prompt_balance,
        },
        "total_episodes": 0, "total_successes": 0, "success_rate": None, "tasks": [],
    }
    started = time.monotonic()

    def save_result():
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        tmp = result_path.with_name(result_path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, result_path)

    total_episodes = total_successes = 0
    try:
        save_result()
        for task_id in tqdm.tqdm(task_ids):
            old_episodes, old_successes = total_episodes, total_successes
            total_episodes, total_successes = run_task(
                cfg, task_suite, task_id, vla, resize_size, processor, action_head,
                proprio_projector, None, total_episodes, total_successes, log_file,
            )
            episodes = total_episodes - old_episodes
            successes = total_successes - old_successes
            result["tasks"].append({
                "task_id": task_id, "task": task_suite.get_task(task_id).language,
                "episodes": episodes, "successes": successes,
                "success_rate": successes / episodes if episodes else None,
            })
            result.update(total_episodes=total_episodes, total_successes=total_successes,
                          success_rate=total_successes / total_episodes if total_episodes else None)
            save_result()
        result["status"] = "completed"
        save_result()
        label = "Quantized Eval" if quantized else "Float Eval"
        rate = total_successes / total_episodes if total_episodes else 0.0
        print(f"[{label}] Total episodes: {total_episodes}, successes: {total_successes}, "
              f"success_rate: {rate:.4f} ({rate*100:.1f}%)", flush=True)
        print(f"[LIBERO] Results: {result_path.resolve()}", flush=True)
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_result()
        raise
    finally:
        if log_file:
            log_file.close()

