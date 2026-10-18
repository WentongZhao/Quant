# Server quantization integration overlays

Captured on 2026-10-18 from the quantization and robot evaluation working copies on the server. The active Quant framework, unified evaluation adapters, configs, tests and launchers are committed at the repository root. These overlays preserve alternate OpenVLA, SimplerEnv and VLA-Pruner adaptations without replacing the active implementation.

Each manifest entry records the upstream URL, exact base commit, changed tracked paths, included untracked source files and patch SHA-256. Patches include tracked changes and new source/configuration/documentation files. Model weights, datasets, rollouts, experiment results, caches, compiled packages, and dated before-edit backups are excluded. Video understanding repositories are excluded.

To restore an overlay, clone its upstream repository into a separate directory, check out the recorded base commit, then run:

```bash
git apply --check /path/to/Quant/server_overlays/NAME.patch
git apply /path/to/Quant/server_overlays/NAME.patch
```

Restore the matching SimplerEnv-OpenVLA overlay in a separate checkout and set `SIMPLERENV_NATIVE_ROOT` to that checkout for native evaluation/tests. Local absolute paths in scripts and configs describe the source server and may need adjustment elsewhere. Preserving an overlay does not claim its policy or quantization experiments have been validated.
