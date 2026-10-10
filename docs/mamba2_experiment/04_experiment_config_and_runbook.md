# Experiment configuration and runbook

Status: launch commands and CUDA preflights verified 2026-10-05. Full-run status and updates are tracked in `07_training_and_results.md`.

## Environment and upstream packages

| Component | Verified value |
|---|---|
| OS/runtime | WSL2, Ubuntu 24.04.5 |
| Python / manager | Python 3.13.15 in the project `.venv`; uv 0.12.18 |
| PyTorch | 2.10.0+cu130; CUDA runtime 13.0 |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16,303 MiB reported |
| NVIDIA driver | 610.60 |
| CUDA compiler | `nvcc` is absent |
| bfloat16 support | CUDA device reports support; this experiment remains `32-true` because only FP32 was exercised end to end |
| Mamba-2 | Official `state-spaces/mamba` commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`; package version `2.3.2.post1` |
| Causal convolution | Official `Dao-AILab/causal-conv1d` 1.7.0 wheel matched to Linux x86_64, Python 3.13, Torch 2.10, CUDA 13, C++11 ABI true |

The project pins the full Mamba Git commit and wheel URL in `pyproject.toml` / `uv.lock`. Mamba is installed from the official source with its selective-scan CUDA extension left off, so this setup does not need `nvcc`; official Mamba-2 Triton kernels and causal-conv1d were tested on the GPU. Upstream installation and API references: [official Mamba repository](https://github.com/state-spaces/mamba), [Mamba-2 module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py), [causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases), and [PyTorch previous-version CUDA wheel matrix](https://pytorch.org/get-started/previous-versions/).

Install/update the locked environment with:

```bash
uv sync
```

## Presets

- `configs/mamba2_prose.yml` is the default controlled comparison. It follows the paper prose: 240 epochs, 12 warmup epochs, four predictor layers, weight decay 0.04, and effective aggregate batch 256. Per-source microbatch is 8 with eight-step gradient accumulation across four sources. The microbatch/accumulation choice is for the 16 GB GPU; it preserves the nominal effective aggregate batch.
- `configs/jepa_vit_baseline.yml` matches the prose-default data mixture, masks, predictor, effective batch, and training schedule with the original ViT encoder.
- `configs/mamba2_table1.yml` preserves Table I's alternative: 3000 epochs, 150 warmup epochs, effective aggregate batch 128, two predictor layers, two predictor masks, context scale `[0.85,1.0]`, and final weight decay 0.4.
- `configs/mamba2_smoke.yml` is a one-source bounded fixture. `--smoke-batch-size` can override its default of one for a memory measurement.

The paper does not resolve its prose/Table I differences. Treat the prose preset as a first comparison choice, not as a claim that Table I is wrong. The 5G CSI candidate remains excluded pending provenance clarification; the four configured sources total 24,552 samples.

## Verified preflight and smoke commands

The exact four-source launch command in dry-run mode passed for both Mamba-2 and ViT. It verified all configured HDF5 keys and finite representative samples and produced the same preflight shape `(48, 28, 256)`. The Mamba run's allocated-memory peak was 361,396,224 bytes; the ViT run's was 98,454,016 bytes. These preflight figures do not estimate training memory.

```bash
uv run python src/pretrain.py \
  --config configs/mamba2_prose.yml \
  --device cuda \
  --dry-run
```

All four data sources were listed in run metadata. The output directory was `/mnt/c/Users/danis/Downloads/data2/h5_staging/MultiScaleJEPA/runs/latent-wave/mamba2_prose`, outside the Git root. The ViT dry-run command was:

```bash
uv run python src/pretrain.py \
  --config configs/jepa_vit_baseline.yml \
  --device cuda \
  --dry-run
```

The bounded eight-example CUDA smoke passed through one optimizer/EMA/checkpoint update on actual NRPos Indoor HDF5 samples:

```bash
uv run python src/pretrain.py \
  --config configs/mamba2_smoke.yml \
  --device cuda \
  --output-dir ../runs/latent-wave/mamba2_smoke_batch8 \
  --smoke \
  --smoke-batch-size 8
```

It reported finite loss `1.041164`, nonzero student and predictor gradients, a changed target parameter after EMA momentum `0.9`, and peak PyTorch CUDA allocation `1,407,824,896` bytes. This is one step on one dataset; it does not measure a full training epoch.

The regression suite passed:

```bash
uv run pytest -q
```

Result: 12 passed, including real CUDA Mamba-2 tests, the HDF5-backed JEPA loss/backward test, and the downstream shared-split manifest check.

## Full launch, outputs, and resume

The seed-0 run uses the supported seed/output-directory arguments below; remaining matched seeds use the same CLI override. Create the log directory before starting `tee`:

```bash
uv run python src/pretrain.py \
  --config configs/mamba2_prose.yml \
  --device cuda \
  --seed 0 \
  --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0
```

The explicit seed-specific directories are under `../runs/latent-wave/full_runs/`, outside the Git root. Each saves `resolved_config.yml`, `run_metadata.json`, `train.jsonl`, `train_console.log`, `latest.pt`, `best.pt`, and periodic `epoch-*.pt` files. In-epoch terminal progress is printed about every 5% of optimizer updates; epoch JSONL records include architecture, seed, epoch/total, step/total, per-source and mean train/validation losses, learning rate, EMA momentum, elapsed time, ETA, peak allocated/reserved GPU memory, throughput, and source example contributions. Checkpoints include architecture, seed, resolved config, Git revision/dirty state, epoch/step, optimizer/scheduler, RNG and sampler state, and cumulative elapsed time.

The 5% label-free validation holdout is made separately within each configured pretraining source. Sources are max-size-cycled and weighted equally. Training loaders drop incomplete microbatches and the final incomplete accumulation window is trimmed, so every optimizer update uses 8 microbatches × 4 sources × 8 examples = 256 examples. The actual per-epoch step count is recorded in `run_metadata.json`.

Resume from the last completed epoch with:

```bash
uv run python src/pretrain.py \
  --config configs/mamba2_prose.yml \
  --device cuda \
  --seed 0 \
  --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0 \
  --resume ../runs/latent-wave/full_runs/mamba2_prose_seed0/latest.pt
```

For the seed-specific run, resume with `--resume ../runs/latent-wave/full_runs/mamba2_prose_seed0/latest.pt` and retain the same `--seed 0` and output directory. ViT uses the same arguments with `configs/jepa_vit_baseline.yml` and `../runs/latent-wave/full_runs/vit_prose_seed0/`. Seeds 1 and 2 use `--seed 1` / `--seed 2` and matching `_seed1` / `_seed2` directories. Runs are GPU-sequential.

SIGINT/SIGTERM requests are handled after the current epoch is checkpointed. Checkpoints include model, optimizer, scheduler, sampler generator, Python/NumPy/Torch/CUDA RNG, epoch, step, validation score, and resolved config. Use `best.pt` for the selected validation-loss checkpoint; do not select a checkpoint using downstream test scores.

The ViT baseline launch uses `configs/jepa_vit_baseline.yml`; no CPU fallback is provided for Mamba-2. Mixed precision is configurable as `bf16-mixed`, but the supplied presets use FP32 because a full JEPA bfloat16 smoke was not run. For live visible output, run each command in the foreground as `set -o pipefail; uv run python -u ... 2>&1 | tee -a <run-dir>/train_console.log` after creating the run directory.

## Performance follow-up

The bidirectional Mamba-2 preset is a principled first candidate, not a measured improvement. It holds width, depth, predictor, masks, source mixture, and effective batch to the ViT comparison, but has 38.8% more total parameters. After the initial three-seed comparison, any optimization should be a separate validation-only experiment: vary one documented setting at a time (for example `d_state` 64/128, `expand` 1/2, or mask geometry/count), and use only training/validation data to choose it. If downstream-probe settings need tuning, use inner cross-validation inside each training split; keep the persisted test indices untouched until final reporting. No tuning sweep or superiority result is claimed here.

## Current limitations

- Live run status is tracked in `07_training_and_results.md`. At the 2026-10-08 01:47 MDT update, Mamba-2 seed 0 had completed 40/240 epochs and was continuing; ViT seed 0 remained at its preserved epoch-1 checkpoint behind the sequential handoff. No downstream evaluation had run. Both runs remain partial.
- The batch-8 smoke confirms a bounded update fits comfortably in the current GPU process, but full-run memory and throughput can differ with each source's channel count and validation pass.
- The GPU preflight was against CUDA 13.0 / Torch 2.10 and the pinned wheel; another OS, Python ABI, GPU, or Torch build needs a matching dependency pin and a new real CUDA gate.
- The CSI source values are real and nonnegative, but metadata does not settle magnitude versus power. Inputs are consumed exactly as the existing caches store them.
- The pretraining validation split is made after loading already-normalized caches. Their HDF5 attributes do not prove whether upstream normalization statistics were fit on the later train partition alone; this inherited preprocessing fit-scope is unresolved. The new downstream embedding standardizers are fit on training splits only.
