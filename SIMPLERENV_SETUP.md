# SimplerEnv environment on this server

Prepared 2026-10-12 in a separate Conda environment, `quant-simplerenv`.
The original `quant-openvla-oft` environment was cloned offline; its package
list is unchanged. Python downloads used the Tsinghua PyPI mirror. The missing
system Vulkan loader (`libvulkan1`) was installed from the Tsinghua Ubuntu mirror.

## Usage

```bash
conda activate quant-simplerenv
cd /root/autodl-tmp/Quant
python evaluate.py --list
```

The environment's pip.conf defaults to https://pypi.tuna.tsinghua.edu.cn/simple.
Use HF_ENDPOINT=https://hf-mirror.com when downloading model weights.

## Versions

- Python 3.10 (cloned from the OFT environment)
- PyTorch 2.2.0, torchvision 0.17.0
- Custom OFT Transformers 4.40.1, Flash Attention 2.5.5
- NumPy 1.26.4, SciPy 1.15.3
- SAPIEN 2.2.2, Gymnasium 0.29.1, Ruckig 0.12.2, Rtree 1.4.1
- mediapy 1.2.0
- Editable SimplerEnv and ManiSkill2_real2sim from
  /root/autodl-tmp/zwt/SimplerEnv-OpenVLA-OFT

## Validation completed

- pip check: no broken requirements.
- SimplerEnv import and registration of 25 task names.
- Google Robot coke-can and WidowX spoon-on-towel scenes: reset, 3 physics
  steps, nonempty RGB observations, PNG output, clean process exit.
- Migrated OFT main_quant.py --help and lazy OFT runtime imports.
- No model weights were loaded for these checks: these are simulator/dependency
  checks, not policy success-rate experiments.

Logs, full package lock, scripts and rendered PNGs:
`runs/simplerenv-setup-20261012/`.
`validation.exitcode` is 0 and validation.log ends with VALIDATION_COMPLETE.
The missing DISPLAY warning is expected for headless Vulkan rendering; both
scenes rendered successfully without an X server.

## Remaining before policy evaluation

Download a checkpoint compatible with the selected SimplerEnv robot/task,
including its required action normalization statistics and OFT action head.
The currently downloaded LIBERO Spatial checkpoint is not a substitute for a
Bridge/WidowX or Google Robot policy. Then run a policy smoke test before a
full screen job. SimplerEnv quantization routing is still pending validation.

## OpenVLA Bridge FP evaluation

The native OpenVLA path uses `bridge_orig` action statistics and the four
WidowX tasks in `scripts/bridge.sh`: carrot, cube stack, spoon, eggplant.
The JSON configs preserve the robot poses, overlays, 5 Hz control, 500 Hz
simulation and 60/120-step limits; the full protocol has 24 episodes per task.
The smoke config runs episode 0 once per task, with the same step limits.

```bash
conda activate quant-simplerenv
bash scripts/start_simplerenv_screen.sh /root/autodl-tmp/models/openvla-7b
```

The screen job runs smoke first and starts 96-episode FP evaluation only after
all four smoke episodes finish without runtime errors. Task failure is a valid
result and does not stop evaluation. Each phase writes `results.json` with
per-task success flags and totals, updated atomically at task boundaries.
`run.json` tracks worker status; logs, videos and action plots are retained.
Errors mark the result failed and prevent progression. This launcher does not
automatically resume an interrupted run; use a new output directory.

This is BF16 policy inference with Flash Attention 2 where available, not
quantized evaluation. No SimplerEnv quantization support is claimed yet.

Video export also requires `ffmpeg` on PATH. On the current server it comes
from `imageio-ffmpeg==0.6.0`, installed via the Tsinghua PyPI mirror, with
`quant-simplerenv/bin/ffmpeg` linked to the bundled binary. Install a system
ffmpeg or expose this binary before launching a run.
