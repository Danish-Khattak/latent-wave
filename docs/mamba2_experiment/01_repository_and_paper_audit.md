# Repository and paper audit

Audit date: 2026-10-05. This report was written before changing model code.

## Verified project state

- Git root: `latent-wave/`; branch: `master`; remote: `origin` at `Danish-Khattak/latent-wave`; the worktree was clean at the start of this task.
- The HDF5 files are in the workspace sibling `data/` directory, outside Git. No dataset file was edited.
- The README contains only the project heading. The working description is therefore in `configs/jepa.yml`, the `src/` tree, and the attached six-page paper.
- The repository is a `uv` project (`pyproject.toml`, `.python-version`, `uv.lock`). It requires Python 3.13+, pins a CUDA 12.6 PyTorch index, and resolves Torch 2.10.0+cu126. There are no existing tests.

## Code map

| Role | Current implementation |
|---|---|
| Per-channel patch embedding | `src/models/components.py:PatchEmbed` reshapes `(B,C,H,W)` to `(B*C,1,H,W)`, applies one shared `Conv2d(1,D,kernel=16,stride=16)`, then restores `(B,C,P,D)`. At 224×224 this gives 14×14 = 196 tokens per channel. |
| Positions and context/target encoder | `src/models/encoder.py:Encoder` adds fixed 2-D sin/cos positions to the patch grid, gathers context indices per channel, flattens channel-major then patch-major to `(B,C*P,D)`, applies eight ViT blocks, then LayerNorm. Positions encode patch location but not antenna number. |
| Masks and collation | `src/collators/multi_block_mask.py` samples the same number of indices for every item in a batch by truncating masks to the smallest sampled length. The default region masks are row-major because they use `nonzero` on a flattened grid. `RandomMask` retains random-permutation order. `src/models/utils.py:apply_masks` applies the same spatial indices to every channel. |
| Predictor | `src/models/predictor.py:Predictor` projects context tokens from 256 to 128 dimensions, adds target positional embeddings and learned mask tokens, runs four transformer blocks in the main config, and projects back to 256. It receives channel count and mask indices so target ordering follows the collator. |
| JEPA objective and teacher | `src/models/jepa.py:JEPA` runs the context encoder on visible tokens, the target encoder on the full input, LayerNorms target latents, gathers target indices, and computes MSE. The target starts as a deep copy and has parameter gradients disabled. Existing EMA zips parameters without name checks, ignores buffers, and runs from `on_train_batch_end`; the inherited `.train()` can put the target back in training mode. |
| Pretraining data and sampling | `src/datasets/pretrain.py:PretrainCache` loads a configured HDF5 key as float32. `src/pretrain.py` builds one loader per configured dataset and combines them with Lightning `CombinedLoader(mode="max_size_cycle")`. `JEPA.training_step` averages per-source losses each combined step, so smaller sources are cycled and each source contributes equally per update. The current channel sampler chooses `C′` from 2..C when C>1, while the paper says 1..C. |
| Schedules and checkpoints | `src/models/jepa.py` uses AdamW plus warmup/cosine LR and cosine weight-decay schedules. Lightning saves epoch checkpoints every five epochs with `save_top_k=-1`; the CLI has no explicit resume, dry-run, or bounded smoke mode, and does not save a selected/best checkpoint. |
| Downstream entry points | `src/pretrain.py` evaluates RF classification, positioning, and two DeepMIMO labels during JEPA validation. `src/finetune.py` is a separate trainable-head/finetuning CLI. `src/evaluation/` contains logistic-regression and linear-regression embedding probes. `src/data.py` supports sample-level random or stratified splits. |
| Existing paths and ignore rules | `configs/jepa.yml` points at `../preprocessed_datasets/*.h5`, while the inspected workspace data are under `../data/{pretraining,downstream}/`. `.gitignore` excludes run/checkpoint directories but has no general in-repository HDF5 guard. |

## Paper details and unresolved choices

The attached paper describes a shared per-channel 16×16 patch projection, 2-D sinusoidal positions, stochastic channel selection, multi-block context/target masks, a full-input EMA target, a smaller predictor, LayerNorm on targets, and masked-token MSE. It reports a 224×224 input, 256-dimensional encoder, eight layers, and eight heads. Frozen downstream embeddings are mean-pooled and standardized from training data only; classification uses logistic regression and mean-per-class accuracy, while positioning uses linear regression and mean Euclidean error.

The paper conflicts with itself. The prose specifies 240 epochs, 12 warmup epochs, batch size 256, a four-layer predictor of about 880K parameters, and weight decay 0.04. Table I specifies 3000 epochs, 150 warmup epochs, batch size 128, a two-layer predictor of about 490K parameters, and final weight decay 0.4. The repo's original `jepa.yml` aligns with the prose for epochs, warmup, predictor depth, EMA range, and weight decay; its per-source batch size is 64, which becomes 256 input examples per combined update across four sources. It does not settle the duration/batch/predictor parameter-count/weight-decay conflicts. `jepa.yml` also uses six predictor masks and a fixed context scale of 1.0, unlike Table I's two masks and the paper's broader context-scale interval. These values remain explicit in named presets.

The code currently evaluates supervised downstream labels during JEPA validation. That is not part of JEPA loss training, but it violates the requested separation and should not be used in the new controlled pretraining path. The existing evaluation probes split the full dataset repeatedly rather than consuming one persisted set of indices; the positioning evaluator uses only seed 0. For DeepMIMO, a `scenario` identifier exists, but the paper reports sample-level 80/20 splits and does not specify group splits. The outdoor positioning cache has a `source_file` identifier; its grouping semantics need to be checked before considering group-based splitting. The implementation should preserve the paper's sample-level comparison protocol and expose the exact split indices for cross-architecture comparisons.

## Initial decisions for the experiment

1. Keep the current ViT encoder selectable and make the controlled Mamba-2 comparison share the same patch projection, positions, masks, predictor, target normalization, objective, source sampling, and splits.
2. Use the prose-aligned values as the default (240 epochs, 12 warmup epochs, 256 examples per optimizer update, predictor depth 4, weight decay 0.04) while preserving an explicit Table I alternative. The GPU-friendly matched presets use per-source microbatch 8 and accumulate eight updates to reach an effective aggregate 256; the Table I preset accumulates four to reach 128. Do not describe either paper interpretation as uniquely correct.
3. Correct the new experiment's data paths to the actual sibling `data/` tree, exclude downstream labels from pretraining, save fixed split indices, and report parameter/memory measurements instead of assuming model-size parity.
4. Use the official `state-spaces/mamba` Mamba-2 module pinned to commit `e9594ce1c732d97440f0332fdc43170a2294dbfa`, with the matching official causal-conv1d CUDA 13 / Torch 2.10 wheel. The RTX 5070 Ti is Blackwell and `nvcc` is absent, so the core Mamba package is installed without compiling selective-scan CUDA extensions. The real official Triton/causal-conv1d forward/backward and full JEPA smoke subsequently passed on CUDA.

## Readiness update

The matched prose-default dry run opened all four configured sources and completed a real CUDA preflight (`48 × 28 × 256` predictor output). The bounded CSI smoke and the CUDA regression suite also passed. Final dependency, hardware, command, and measured parameter/memory details are in `04_experiment_config_and_runbook.md`; no full pretraining or downstream probe was run.

## Performance aim

The user's aim is for the Mamba-2 model to do better than the previous encoder. No performance result can be promised without training and the four downstream evaluations. The first run should preserve the original data and JEPA behavior so an improvement is interpretable. After that, tune only documented, configurable choices (such as Mamba state size, expansion, directionality, mask geometry/count, and training budget) using a validation protocol that does not select from downstream test data. Compare both architectures on shared seeds and split indices and report each task because the paper itself finds task-dependent mask effects.
