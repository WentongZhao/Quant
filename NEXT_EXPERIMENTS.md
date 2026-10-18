# Next experiments

## Hypotheses
1. The OFT LIBERO FP result (58.0%) has a checkpoint/inference/protocol mismatch to resolve before interpreting quantization against the published 97.6% Spatial reference.
2. The migrated SimplerEnv adapter can evaluate original OpenVLA on the Bridge/WidowX tasks with its native action statistics.
3. W8A8 can retain policy success better than W4A4 under an identical task protocol; low-bit weight and activation effects should be separated if needed.

## Minimal setup and benchmarks
- Keep OpenVLA-OFT/LIBERO and original OpenVLA/SimplerEnv results in separate model/checkpoint groups.
- Diagnostic LIBERO run: all 10 Spatial tasks, first 5 initial states per task, seed 7; compare the direct native evaluator and unified FP route.
- SimplerEnv smoke: one rollout per each of the 4 WidowX tasks, then the migrated bridge.sh protocol (24 episodes/task, 96 total), fixed robot poses, control frequency 5 Hz, simulator 500 Hz; 60-step horizon except eggplant at 120.
- Full LIBERO confirmation after fixes: 50 episodes/task, 500 total per precision.

## Baselines and main experiments
P0: Diagnose OFT FP using identical checkpoint, action head, proprio projector, normalization key, preprocessing, images and action chunk size. If both entrypoints are poor, inspect the shared runtime and compare attention implementations and upstream source versions. Do not attribute the FP gap to quantization.
P1: Original OpenVLA BF16 on SimplerEnv WidowX: spoon/towel, carrot/plate, stacking cubes, eggplant/basket. Use bridge_orig statistics from the downloaded checkpoint, after verifying the model config. Save per-task successes and failures.
P2: Implement/validate original OpenVLA quantization dispatch and Bridge calibration inputs; run FP, W8A8 and W4A4 with identical task settings and seed. LIBERO calibration inputs are not substituted for Bridge inputs.
P3 (conditional): W4A16 and W16A4 isolate weight versus activation error. Keep SQ alpha, calibration sample identities, symmetry and layer scope fixed. Only then consider calibration-size or alpha ablations.

## Metrics
Quality: task and aggregate success counts, total valid episodes, absolute success-rate loss versus each checkpoint's own FP baseline; distinguish runtime errors from task failures.
Systems: peak GPU/CPU memory, model inference latency after warmup (CUDA synchronized), simulator time and end-to-end time separately. Current fake quantization is not an integer-kernel acceleration benchmark.

## Implementation touchpoints
Quant/evaluation/dispatch.py and simplerenv.py; migrated policy adapter; Bridge calibration loader; common results conversion including validated full/partial success semantics.

## Failure modes and decision rules
- Missing robot statistics, action head mismatch, wrong image crop/rotation, gripper/action scaling, incompatible attention masks, missing assets and runtime exceptions block full evaluation.
- Simulator smoke success is not policy correctness. Require valid nontrivial model actions and error-free policy rollouts before the 96-episode SimplerEnv baseline.
- If FP remains far below the matching published setting, resolve baseline mismatch before further quantization sweeps.
- Predeclare a useful W8A8 target of <=3 percentage points absolute loss against a validated FP baseline; expand seeds/episodes before claiming differences of a few points.
- If W4A4 loses tens of points, test mixed precision before spending on larger uniform W4A4 sweeps.

## Downloads
Original OpenVLA: openvla/openvla-7b, pinned revision and hashes in runs/openvla-base-download/status.json. Mirror: https://hf-mirror.com. Destination: /root/autodl-tmp/models/openvla-7b. This is the original OpenVLA, not an OFT Bridge checkpoint.
