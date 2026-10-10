# LatentWave JEPA

This repository contains the LatentWave JEPA implementation and a selectable official Mamba-2 encoder experiment. The ViT encoder remains available for a matched baseline. HDF5 data and generated checkpoints/logs live outside the Git root.

## Status

The official Mamba-2 CUDA forward/backward, 12 regression tests, four-source dry run, and an eight-sample HDF5-backed CUDA optimizer smoke passed. Full pretraining and downstream evaluation have not been run, so there are no performance results yet.

## Start here

- [Repository and paper audit](docs/mamba2_experiment/01_repository_and_paper_audit.md)
- [HDF5 inventory and mapping](docs/mamba2_experiment/02_dataset_inventory.md)
- [Mamba-2 implementation](docs/mamba2_experiment/03_mamba2_implementation.md)
- [Environment, verified commands, and runbook](docs/mamba2_experiment/04_experiment_config_and_runbook.md)
- [Downstream evaluation protocol](docs/mamba2_experiment/05_downstream_evaluation.md)
- [Implementation and test log](docs/mamba2_experiment/06_implementation_log.md)

To reproduce the bounded CSI smoke on the verified WSL CUDA setup:

```bash
uv sync
uv run python src/pretrain.py \
  --config configs/mamba2_smoke.yml \
  --device cuda \
  --output-dir ../runs/latent-wave/mamba2_smoke_batch8 \
  --smoke \
  --smoke-batch-size 8
```

Review the runbook before starting the 240-epoch prose-default experiment.
