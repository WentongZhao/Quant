# Unified evaluation

`evaluate.py` is the public entrypoint in **Quant**. Model backend, benchmark and
precision mode are independent arguments. FP and quantized OpenVLA-OFT policies
use the same LIBERO task loop, initial states, action head and preprocessing.

```text
Quant/
  evaluate.py                    public entrypoint
  evaluation/
    cli.py                       CLI + YAML configuration, validation
    dispatch.py                  supported combinations and isolated workers
    libero.py                    common FP/quant task loop and JSON metrics
    simplerenv.py                bridge to migrated SimplerEnv adapters
  configs/evaluation/            reproducible experiment settings
  scripts/run_libero_comparison.sh
  scripts/start_libero_screen.sh
  tests/                         routing and metric regression checks
  runs/<run>/                    configuration, logs, metrics and rollout videos
  qvla/                          existing calibration and quantization methods
  prismatic/                     existing OpenVLA-OFT model implementation
  experiments/robot/             existing model loading and robot adapters
  main_quant.py                  internal/legacy quantization worker
```

The existing model and quantization source trees remain authoritative. This
incremental restructuring does not duplicate them or move datasets/checkpoints.
`main_quant.py` and `quant1.sh` remain usable for existing workflows. New
experiments should use `evaluate.py`; the migrated `zwt` copies are not competing
public entrypoints for this framework.

## Supported routes

| Model | Benchmark | Mode | State |
|---|---|---|---|
| OpenVLA-OFT (L1 head) | LIBERO | FP | Unified worker, no calibration or quantizer call |
| OpenVLA-OFT (L1 head) | LIBERO | quant | SmoothQuant, same task loop as FP |
| OpenVLA | SimplerEnv | FP | Bridge to migrated adapter; environment validation needed |
| OpenVLA-OFT | SimplerEnv | FP | Bridge to migrated OFT adapter; environment validation needed |
| OpenVLA-OFT | SimplerEnv | quant | Pending symmetry/config validation; rejected by dispatcher |

`python evaluate.py --list` lists wired routes, not proof that their dependencies
or model weights are installed. The current `quant-openvla-oft` environment is
configured for LIBERO and lacks the SimplerEnv simulator dependencies. A separate
`quant-simplerenv` environment now passes simulator and OFT import checks; see
[SIMPLERENV_SETUP.md](SIMPLERENV_SETUP.md). SimplerEnv
also requires compatible checkpoint normalization statistics; the current
LIBERO Spatial checkpoint alone does not supply those.

The bridge defaults to the sibling `zwt/SimplerEnv-OpenVLA` or
`zwt/SimplerEnv-OpenVLA-OFT` directory. Override with `--simplerenv_root` and pass
`--tasks_json`. Task layout/episode counts follow the migrated task JSON. An
OpenVLA FP official suite can instead use `--benchmark_subset`. SimplerEnv native
metrics remain in its rollout output; only LIBERO currently writes the common
`results.json` schema. Other models and quantizers are not silently substituted.

## LIBERO experiments

Activate the existing environment, and use EGL for headless rendering:

```bash
conda activate quant-openvla-oft
cd /root/autodl-tmp/Quant
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python evaluate.py --config configs/evaluation/libero_spatial.yaml --mode fp
python evaluate.py --config configs/evaluation/libero_spatial.yaml --mode quant

# Same task subset for a short rendering/inference validation:
python evaluate.py --config configs/evaluation/libero_spatial.yaml \
  --mode fp --eval_task_ids 0 --num_trials_per_task 1
```

CLI options override YAML, which overrides built-in defaults. YAML relative paths
are resolved against the configuration file. Unknown keys are errors. Use
`--dry_run` to inspect the planned worker and configuration without loading a
model or writing a run directory.

The example config contains this server's paths and W8A8 scale file. For W4A4,
change **both** bit widths and `--scale_path`. The comparison script does this.
FP uses BF16 model weights and Flash Attention 2. Quantization uses simulated
SmoothQuant, with symmetric weights and asymmetric activations by default;
it is an accuracy evaluation, not a packed integer-kernel speed benchmark.
The default target is the language model; vision, action head and proprio
projector retain their existing floating-point implementations. KV stays at 16.

`--run_process` is optional when recalibrating quant mode; an existing scale file
is required otherwise. FP ignores calibration data and scale settings and rejects
`--run_process`. Changing the checkpoint or calibration settings requires matching
scales. The original calibration stage still constructs inputs when reusing scales.

## Detached comparison

```bash
bash scripts/start_libero_screen.sh
# Prints the new screen name and run directory.
screen -r <printed-name>
# Detach without stopping: Ctrl-A, then D.

# Optional smoke run (one task, one trial per precision):
CASES='fp w8a8' NUM_TRIALS=1 EVAL_TASK_IDS=0 \
  SCREEN_NAME=libero-smoke bash scripts/start_libero_screen.sh
```

The formal default sequence is FP, W8A8, W4A4, each with 10 tasks x 50 trials,
seed 7, two images, proprioception, L1 action head and the same initial states.
Scales are reused from the previous 256-sample, prompt-balanced,
proprio-stratified calibration. A lock prevents overlapping comparison scripts.
SSH disconnects leave screen running; a server shutdown or GPU migration does not.

Each comparison directory contains `driver.log`, package/GPU metadata, one console
log per precision and an `exitcode` file written when the script finishes. Each
precision subdirectory contains:

```text
run.json                 requested settings, worker command, exit status
backend_config.json      resolved worker configuration
results.json             task results, aggregate rate, dtype, FA2, precision
logs/                    native LIBERO text logs
rollouts/                native episode videos
```

LIBERO JSON progress updates after each task; episode progress is in the logs.
Runtime episode errors fail the unified run instead of counting as policy task
failures. The older direct evaluator retains its previous default behavior.
Output directories containing a run are not overwritten. The previous
`smoothquant_logs`, scales and transferred workspace are preserved.

## Validation

```bash
python -m unittest discover -s tests -p 'test_unified_evaluation.py' -v
```

The tests exercise FP bypass, the quantizer's returned model, configuration
precedence, supported routes and completed/failed metric records. Actual GPU
smoke runs are also required before launching the full comparison.
