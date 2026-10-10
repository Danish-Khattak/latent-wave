# Partial-checkpoint downstream evaluation — 2026-10-09

## Status and checkpoint selection

Mamba-2 seed 0 was stopped at an epoch boundary to prioritize the requested downstream evaluation. The trainer caught `SIGTERM`, completed epoch 163, evaluated validation, saved its checkpoints, wrote `status=stopped_after_epoch`, and exited normally. The run is **partial: 163/240 epochs**.

At that stop, both `latest.pt` and validation-selected `best.pt` contain human epoch **163** (checkpoint epoch field 162), global step **34,882**, mean training JEPA loss **0.231739416**, and fixed-protocol validation JEPA loss **0.270749662**. Epoch 163 improved on epoch 162's validation loss of 0.272443010. The checkpoint includes the target encoder, optimizer, schedulers, sampler/RNG states, resolved configuration, validation protocol, and code revision. The immutable evaluation copy is `../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_seed0_best_epoch163.pt`, SHA-256 `b89ddd3fbac9b397f4ab246c393f4eeb1af50f1421f2873d05ac8a1198d704ad`.

The original training log and checkpoints were preserved. Resume Mamba-2 seed 0 with:

```bash
cd /mnt/c/Users/danis/Downloads/data2/h5_staging/MultiScaleJEPA/latent-wave
set -o pipefail
./.venv/bin/python -u src/pretrain.py \
  --config configs/mamba2_prose.yml \
  --device cuda \
  --seed 0 \
  --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
  --resume ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/latest.pt \
  2>&1 | tee -a ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/train_console.log
```

Use `set -o pipefail` around the command when preserving the process exit status through `tee`.

## Evaluation protocol

The partial checkpoint is a full JEPA checkpoint with `target_encoder.*` weights. `src/evaluate_linear.py` checks the requested architecture, requires CUDA for Mamba-2, and does not require 240 completed epochs. CUDA availability and the NVIDIA GeForce RTX 5070 Ti were confirmed. The four HDF5 paths existed with the expected sample/label datasets; the evaluator opens them with HDF5 read mode. No HDF5 source file was written.

The evaluator used the frozen target encoder, mean pooling over its token dimension, and a sklearn pipeline that fits `StandardScaler` on each training split only. It uses sample-level 80/20 splits for seeds 0, 1, and 2, persisting the indices for reuse by later architectures. Classification splits are stratified by class; the positioning split is random and unstratified. Metrics are mean-per-class accuracy (percent) and mean Euclidean positioning error (metres). The held-out downstream scores are reported only; neither scores nor labels were used for JEPA checkpoint selection or hyperparameter tuning.

Task IDs and inputs were `rfs` / `rfs.h5`, `pos` / `nrpos-outdoor.h5`, `deepmimo-beam` / `deepmimo.h5`, and `deepmimo-los` / `deepmimo.h5`. All used downstream split seeds 0, 1, and 2. The split manifests are in `../runs/latent-wave/shared_splits/controlled_seed0_fixed_validation/`.

Exact execution command:

```bash
cd /mnt/c/Users/danis/Downloads/data2/h5_staging/MultiScaleJEPA/latent-wave
set -o pipefail
bash ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/run_evaluation.sh \
  2>&1 | tee -a ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/evaluation_console.log
```

The script records the supported per-task CLI invocations, using `--device cuda --batch-size 64 --workers 2 --seeds 0 1 2`. The resolved configuration was extracted from the checkpoint to `../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/resolved_config.json`; full environment, paths, protocol, and checkpoint hashes are in `partial_evaluation_protocol.json`. Software was Python 3.13.15, PyTorch 2.10.0+cu130, CUDA 13.0, h5py 3.15.1, NumPy 2.3.5, and scikit-learn 1.8.0.

## Measured results

The ± values below are standard deviations across the three **downstream split seeds for one pretrained checkpoint**. They are not pretraining-seed variability.

| Task | Metric | Mamba-2 seed 0, checkpoint epoch 163 | LatentWave Region | LatentWave Frequency | WavesFM | Supervised |
|---|---|---:|---:|---:|---:|---:|
| RF signal classification | Mean-per-class accuracy (%) ↑ | 70.89 ± 1.14 | 80.9 ± 2.94 | 66.1 ± 3.10 | 80.3 ± 0.73 | 86.5 ± 1.69 |
| 5G NR positioning | Mean positioning error (m) ↓ | 2.429 ± 0.015 | 2.54 ± 0.009 | 2.32 ± 0.023 | 2.77 ± 0.037 | 0.71 ± 0.001 |
| DeepMIMO beam prediction | Mean-per-class accuracy (%) ↑ | 49.35 ± 0.87 | 51.6 ± 0.35 | 63.1 ± 0.45 | 51.2 ± 0.36 | 88.9 ± 0.50 |
| DeepMIMO LoS/NLoS classification | Mean-per-class accuracy (%) ↑ | 91.94 ± 0.47 | 92.9 ± 0.57 | 93.4 ± 0.26 | 93.4 ± 0.54 | 95.9 ± 0.27 |

The local checkpoint is between the paper's Frequency and Region results for RF classification. Positioning error is below the Region and WavesFM values, above Frequency and Supervised. Beam prediction and LoS/NLoS accuracy are below all four paper references. These are descriptive comparisons, not claims of superiority: the local model is partial, the paper reports mean ± SD over paper pretraining seeds, and the local SD is across split seeds for one checkpoint. The paper setup and these local evaluator settings are not a matched rerun.

The frozen linear evaluator does not emit confusion matrices, so none were generated. Fine-tuning was not launched in this evaluation pass.

## PNG figures and result files

All artifacts are outside Git under `../runs/latent-wave/`:

- Paper Table II plus the local partial Mamba-2 results: `comparison/partial_checkpoint_e163_fixed_validation/paper_results_comparison.png`
- Local downstream task scores: `comparison/partial_checkpoint_e163_fixed_validation/comparison.png`
- Validation JEPA loss curves: `comparison/partial_checkpoint_e163_fixed_validation/pretraining_validation_loss.png`
- Task completion/status panels: `comparison/partial_checkpoint_e163_fixed_validation/downstream_task_status.png`
- CSV/Markdown/JSON comparison tables are beside those PNGs; per-task score JSONs and `evaluation_console.log` are in `results/partial_checkpoint_e163_fixed_validation/`.

![Paper Table II and partial local Mamba-2 results](../../../runs/latent-wave/comparison/partial_checkpoint_e163_fixed_validation/paper_results_comparison.png)

## Partial-checkpoint use and remaining protocol limits

- **Frozen downstream evaluation:** supported and completed from the epoch-163 Mamba-2 checkpoint. Seeds 0, 1, and 2 here are downstream split seeds, not three pretraining seeds.
- **Matched-seed architecture evaluation:** the current checkpoint can serve as Mamba seed 0, and the persisted split manifests make the later ViT probes use the same indices. A matched Mamba-vs-ViT comparison is not currently available: ViT seed 0 has only 1/240 epochs, so it does not match Mamba's 163-epoch budget. Mamba pretraining seeds 1 and 2 are also not complete. The full-run queue's outputs remain separate and were not populated by these diagnostic results.
- **Fine-tuning:** `src/finetune_jepa.py` accepts a full JEPA target-encoder checkpoint without requiring 240 pretraining epochs. Its `--validation-only` mode carves validation from the training partition and can use the saved split manifests; it is safe for a partial-checkpoint validation screen. `--test-only` requires a validation-selected fine-tuning checkpoint and should follow candidate selection. The existing supervised fine-tuning queue gates on all full pretraining runs and frozen probes; it did not run here. No fine-tuning scores are reported.

Supported validation-only example (not launched in this pass):

```bash
cd /mnt/c/Users/danis/Downloads/data2/h5_staging/MultiScaleJEPA/latent-wave
./.venv/bin/python -u src/finetune_jepa.py \
  --task rfs \
  --data ../data/downstream/rfs.h5 \
  --checkpoint ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_seed0_best_epoch163.pt \
  --config configs/mamba2_prose.yml \
  --split-dir ../runs/latent-wave/shared_splits/controlled_seed0_fixed_validation \
  --out-dir ../runs/latent-wave/results/finetune_partial_e163_validation \
  --device cuda \
  --split-seeds 0 1 2 \
  --validation-only
```

The current repo revision is `330e9ef5f6eb69e728ba43b3755036cffdd086cd`; the worktree was already dirty before this pass. The original full-run/evaluation/matched-seed/fine-tuning queues observed their 240-epoch gates and exited or remained gated without launching another GPU job. No unrelated process was terminated, no commit was created, and no push was made.

## Current checkpoint follow-up — 2026-10-09 15:43 MDT

Mamba seed 0 has resumed training since this evaluation. `latest.pt` is now human epoch 164, validation loss **0.271787534**; the selected `best.pt` remains human epoch **163**, validation loss **0.270749662**. Its SHA-256 still matches the immutable evaluation copy above, so the already-recorded four-task scores are from the current selected best. The training process is active at epoch 165; it was not paused or changed for this audit.

No repeat probe was launched because those same held-out task splits have already been scored under the recorded protocol. The existing `paper_results_comparison.png` is the comparison for the measured epoch-163 checkpoint; its status line reflects the run at the time of that evaluation (163/240). Matched ViT evaluation is still unavailable because ViT is at 1/240. Validation-only fine-tuning is technically supported with this partial JEPA checkpoint and saved splits, with test samples excluded during selection; test-only fine-tuning requires a validation-selected fine-tuned `best.pt`. The active fine-tuning queue correctly remains gated and was not started concurrently with Mamba training.
