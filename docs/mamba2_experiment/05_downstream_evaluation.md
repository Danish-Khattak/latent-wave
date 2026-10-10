# Downstream evaluation protocol

The full matched comparison remains gated on the matched 240-epoch pretraining runs. An exploratory frozen evaluation has since been completed for the Mamba-2 seed-0 best partial checkpoint (epoch 163/240); it is documented separately in [08_partial_checkpoint_downstream_evaluation.md](08_partial_checkpoint_downstream_evaluation.md). Do not treat those partial scores as the completed primary comparison.

## Latest run status — 2026-10-09 03:27 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **127/240**, global step **26,980/51,360** (**16/214** optimizer steps), with rolling train loss **0.270056**. The latest completed fixed validation is **0.311929053** from epoch 126; the selected best remains epoch **124** at **0.303736491**. LR is **4.99985e-4**, EMA **0.998101**, elapsed time **129,323 s**, and ETA **116,860 s (~32h28m)**. PyTorch allocation/reservation is **2.23/4.94 GiB**; the RTX 5070 Ti reports **6,813/16,303 MiB** at **15%** utilization. `run_metadata.json` and `latest.pt` confirm 126 completed epochs and step **26,964**; `best.pt` remains selected from epoch 124.

ViT seed 0 remains at its preserved epoch-1 checkpoint. The guarded ViT handoff, seed-0 evaluation, matched-seed, and fine-tuning queues remain alive and gated. The seed-0 evaluator waits for both seed-0 runs to confirm all **240 epochs**, then waits for GPU idle; its results directory contains only `evaluation_console.log`, so downstream scores remain **0/8**. No downstream test result has been used for selection.

## Prior run status — 2026-10-08 23:09 MDT

Mamba-2 seed 0 PID **1717366** is active at epoch **113/240**, global step **23,980/51,360** (**12/214**), with rolling train loss **0.277366**. Latest completed fixed validation is **0.309308745** from epoch 112; it is the new best checkpoint. LR is **5.9588e-4**, EMA **0.997868**, elapsed **114,168 s**, and ETA **130,355 s (~36h12m)**. PyTorch memory is **2.22/4.20 GiB**; the RTX 5070 Ti reports **6,049/16,303 MiB** at **19%** utilization. Run metadata and checkpoint confirm 112 completed epochs/global step **23,968**. ViT seed 0 remains at its preserved epoch-1 checkpoint.

The guarded ViT handoff, seed-0 evaluation, matched-seed, and supervised fine-tuning queues remain alive and gated. The evaluation queue waits for both seed-0 runs to confirm all **240 epochs**, then waits for GPU idle. Results remain **0/8**; no downstream test score has been used.

A source-level audit of `src/evaluate_linear.py` and the running queue confirms task identifiers `rfs`, `pos`, `deepmimo-beam`, and `deepmimo-los`. The first architecture creates each sample-level 80/20 train/test manifest under the shared split directory; the other architecture checks sample count, label hash, and split-seed set before reuse. Frozen probes use seeds 0, 1, and 2, mean-pooled target-encoder tokens, and a `StandardScaler` fitted within the training-only pipeline. Hyperparameters are fixed; the documented frozen-probe protocol has no separate downstream validation partition or tuning step, and test scores are only for final reporting.

## Prior run status — 2026-10-08 18:58 MDT

Mamba-2 seed 0 has completed 98/240 epochs and is in epoch 99 at global step 20,990/51,360 (18/214); latest and selected best checkpoint is epoch 98/global step 20,972 with fixed-validation loss 0.318487. It was stopped at epoch 98 on user request and resumed from that checkpoint with the same prose config. ViT seed 0 remains at its preserved epoch-1 checkpoint, with its sequential resume watcher waiting for Mamba to finish. The guarded seed-0 evaluation queue waits for both complete 240-epoch runs. No downstream task has been evaluated and no test label or score has been used for selection. All four tasks remain not run.

### Queue status update — 2026-10-08 19:07 MDT

The earlier gated processes exited when Mamba was intentionally marked stopped_after_epoch 98; their logs show no downstream tasks or later-seed runs started. After Mamba resumed, the ViT handoff script was corrected to locate its current trainer PID dynamically and relaunched, together with the seed-0 evaluation gate, matched-seed queue, and fine-tuning queue. All four are confirmed waiting; no evaluation has started and the shared split directory/results remain uncreated.

## Prior run-status snapshot

As of **2026-10-08 18:34 MDT**, Mamba-2 seed 0 has completed 97/240 epochs and is in epoch 98 at global step 20,770/51,360; its latest checkpoint is zero-based epoch 96/global step 20,758, and the selected best is human epoch 90 (fixed-validation loss 0.320327). ViT seed 0 remains at its preserved epoch-1 checkpoint, with the sequential resume watcher waiting for Mamba to finish. The guarded seed-0 evaluation queue is waiting for both full 240-epoch runs. No downstream task has been evaluated and no test label or score has been used for selection. All four tasks remain **not run**. The shared split directory has not been created yet; the evaluator creates a per-task 80/20 split manifest on the first Mamba evaluation, and the ViT evaluation reuses it after validating sample count, label hash, and seed set. The frozen evaluator returns aggregate metrics per split and does not export per-example predictions. Supported task identifiers are `rfs`, `pos`, `deepmimo-beam`, and `deepmimo-los`; split seeds are 0, 1, and 2. The three configured downstream HDF5 paths passed a read-permission check without being opened during active pretraining.

The validation-protocol fix uses matching source splits and deterministic masks/channel subsets for the two architectures: source splits are seeded from the same `config.seed` and source order, while `_evaluate` resets CPU/CUDA RNG and each validation collator per source. Thus Mamba's fixed-protocol epochs 20 onward and ViT's fixed-protocol resume from epoch 2 onward can be compared; the preserved legacy-validation segments (Mamba epochs 1–19 and ViT epoch 1) remain separate and are not treated as matched validation points.

## Tasks and protocol

| Task | HDF5 file / keys | Probe and reported metric |
|---|---|---|
| RF signal classification | `../data/downstream/rfs.h5`; `image`, `label` | Frozen target encoder, mean-pooled tokens, logistic regression; mean-per-class accuracy (%) |
| 5G NR outdoor positioning | `../data/downstream/nrpos-outdoor.h5`; `features`, `label` | Frozen target encoder, mean-pooled tokens, linear regression; mean Euclidean error after coordinate de-normalization (m) |
| DeepMIMO beam prediction | `../data/downstream/deepmimo.h5`; `sample`, `label_beam_16` | Frozen target encoder, mean-pooled tokens, logistic regression; mean-per-class accuracy (%) |
| DeepMIMO LoS/NLoS | `../data/downstream/deepmimo.h5`; `sample`, `label_los` | Frozen target encoder, mean-pooled tokens, logistic regression; mean-per-class accuracy (%) |

Features are standardized using only each training split via `StandardScaler` in the sklearn pipeline. Classification retains the repository's `C=5`, `newton-cg`, and 1000-iteration logistic regression setting; score is `balanced_accuracy_score`, equivalent to mean class accuracy over classes represented in that test split. Positioning uses `LinearRegression`, maps normalized predictions back through the recorded nominal `[0,0,0]` to `[80,60,40]` metre range, then averages per-sample Euclidean error.

Each task uses sample-level 80/20 splits with seeds 0, 1, and 2; classification splits are stratified. The first architecture writes exact train/test indices under the shared split directory. The other architecture reads those files and checks label hash, sample count, and seed set before evaluation. These primary splits match the paper's sample-level protocol. DeepMIMO `scenario` identifiers are correlated groups that can occur on both sides of the split, so the primary metric does not claim scenario-held-out generalization. The `source_file` field in RFS/NRPos caches is unique per sample and cannot define repeated capture groups.

## Run probes for both architectures

After pretraining, set checkpoint names to the selected validation-loss checkpoints. Running this function twice uses the same persisted split indices for every task:

```bash
run_probes() {
  arch="$1"
  pretrain_seed="$2"
  config="$3"
  checkpoint="$4"
  if [ "$pretrain_seed" = 0 ]; then
    result_dir=../runs/latent-wave/results/controlled_seed0_fixed_validation
  else
    result_dir=../runs/latent-wave/results/controlled_pretraining_seeds_fixed_validation
  fi
  for task in rfs pos deepmimo-beam deepmimo-los; do
    case "$task" in
      rfs) data=../data/downstream/rfs.h5 ;;
      pos) data=../data/downstream/nrpos-outdoor.h5 ;;
      deepmimo-beam|deepmimo-los) data=../data/downstream/deepmimo.h5 ;;
    esac
    uv run python src/evaluate_linear.py \
      --task "$task" \
      --data "$data" \
      --checkpoint "$checkpoint" \
      --config "$config" \
      --split-dir ../runs/latent-wave/shared_splits/controlled_seed0_fixed_validation \
      --output "$result_dir/${arch}_pretrain_seed${pretrain_seed}_${task}.json" \
      --device cuda \
      --seeds 0 1 2
  done
}

run_probes mamba2 0 configs/mamba2_prose.yml ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/best.pt
run_probes vit 0 configs/jepa_vit_baseline.yml ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val/best.pt
run_probes mamba2 1 configs/mamba2_prose.yml ../runs/latent-wave/full_runs/mamba2_prose_seed1/best.pt
run_probes vit 1 configs/jepa_vit_baseline.yml ../runs/latent-wave/full_runs/vit_prose_seed1/best.pt
run_probes mamba2 2 configs/mamba2_prose.yml ../runs/latent-wave/full_runs/mamba2_prose_seed2/best.pt
run_probes vit 2 configs/jepa_vit_baseline.yml ../runs/latent-wave/full_runs/vit_prose_seed2/best.pt
```

Each JSON records pretraining seed, downstream split-seed scores, mean, sample standard deviation, checkpoint, split file, metric, and feature dimension. Seed-0 outputs are under `../runs/latent-wave/results/controlled_seed0_fixed_validation/`; seed-1/2 outputs are under `../runs/latent-wave/results/controlled_pretraining_seeds_fixed_validation/`. Run the function only for checkpoints with all 240 epochs complete; do not probe partial checkpoints for the primary comparison.

## Comparison table and plot

```bash
uv run python src/summarize_comparison.py \
  --results ../runs/latent-wave/results/*.json \
  --out-dir ../runs/latent-wave/comparison
```

This writes `comparison.csv`, `comparison.md`, and `comparison.png` outside the Git root. The summary first averages the common downstream split-seed scores within each pretrained checkpoint, then gives each pretraining seed equal weight. When multiple pretraining seeds are available, the comparison error bar is the sample SD across those per-checkpoint means; within-checkpoint split-seed SD is reported separately. With a single pretraining checkpoint, the displayed error bar is explicitly labeled as split-seed SD and is not presented as pretraining-seed variability. Keep all model selection and tuning decisions on training/validation data; the held-out downstream test splits are for final reporting only.

## Supervised fine-tuning

The legacy `src/finetune.py` loader does not accept these full JEPA checkpoints. The separate `src/finetune_jepa.py` path loads the JEPA target encoder, reads the same persisted train/test split, and carves validation indices only from the training partition. Its `--validation-only` phase writes validation metrics and selected checkpoints without scoring the held-out test partition; candidate settings can be selected from those results. After selection, `--test-only` loads the existing validation-selected `best.pt` and evaluates the held-out partition once. Classification output includes confusion matrices and per-class accuracy; positioning output includes true/predicted coordinates and per-example errors.

Static review found and fixed a missing resolved checkpoint-path field. Resume and test-only paths also verify the saved split manifest, train/validation indices, pretraining checkpoint, and settings. Syntax compilation passed, but runtime/data smoke and training have not been run. A CLI-help import probe stalled under active-run I/O pressure and was terminated. Fine-tuning remains unrun. A guarded queue at `../runs/latent-wave/full_runs/run_supervised_finetuning_after_matched_probes.sh` is now waiting for all six matched pretraining runs and 24 frozen-probe result files to complete; it will screen learning rates `3e-4` and `1e-4` using validation metrics only, then evaluate the selected candidate once on shared test splits 0, 1, and 2. Keep these metrics separate from frozen-probe results and do not inspect test scores during candidate selection.


## Current evaluation and plot status — 2026-10-09

At the 03:45 MDT snapshot, no complete pretraining checkpoint was available for the planned primary comparison: Mamba-2 seed 0 had completed 127/240 epochs and ViT seed 0 had completed 1/240. No downstream task scores have been produced (0/8 frozen-probe outputs). The four-task status plot therefore shows “not evaluated”; the paper comparison plot contains paper reference bars alongside blank local result slots. These images report status, not local downstream performance. They are in `../runs/latent-wave/comparison/current/downstream_task_status.png` and `../runs/latent-wave/comparison/current/paper_results_comparison.png`.

At the refreshed **04:28 MDT** snapshot, Mamba seed 0 had completed **129/240** epochs and ViT seed 0 remained at **1/240**. The downstream queues are active but gated on both full runs, so local downstream results remain **0/8** and the task status panels still show “not evaluated.”

At **04:47 MDT**, Mamba seed 0 had completed **130/240** and ViT remained at **1/240**. Evaluation gates are still waiting for both full runs; no downstream test splits have been opened and no local scores are available (**0/8**).

At **04:50 MDT**, Mamba seed 0 is still active in epoch **131** (50/214 steps); the latest saved checkpoint is epoch **130**. ViT remains stopped after epoch **1** and is queued behind Mamba. The frozen-probe evaluation script remains gated on both complete 240-epoch runs; downstream results remain **0/8** and no downstream test scores have been accessed. Existing PNG snapshots show only completed pretraining epochs through Mamba 130 / ViT 1.


At **05:05 MDT**, Mamba seed 0 completed epoch **131** and saved the new latest checkpoint; ViT remains at epoch **1/240** and is queued behind Mamba. The seed-0 frozen-probe script remains gated on both complete 240-epoch runs. Results remain **0/8** and no downstream test partitions have been evaluated. The refreshed status and paper-comparison PNGs show Mamba pretraining through epoch 131, ViT through epoch 1, and downstream tasks as not evaluated.


At **05:24 MDT**, Mamba seed 0 completed epoch **132**, produced a new fixed-validation best checkpoint (**0.296659710**), and is continuing. ViT remains at **1/240** and is queued behind Mamba. Frozen-probe evaluation remains gated until both 240-epoch runs complete; local task results remain **0/8**, and held-out downstream test partitions have not been evaluated. The refreshed status and paper-comparison PNGs show the partial pretraining status only.


At **05:42 MDT**, Mamba seed 0 has completed epoch **133/240** and the best fixed-validation checkpoint remains epoch **132** (**0.296659710**). ViT remains at **1/240**, queued behind Mamba. The frozen-probe gate is still waiting for both complete 240-epoch runs; local downstream results remain **0/8**, and no held-out task scores have been evaluated. Refreshed plots show these partial statuses.


At **05:44 MDT**, the seed-0 Mamba run was active at epoch **134**, 30/214 steps. ViT remains stopped after epoch 1 pending the GPU-sequential handoff; evaluation remains gated on both full 240-epoch runs. Local downstream results remain **0/8**, and no downstream test partition has been evaluated.


At **05:46 MDT**, Mamba seed 0 is active in epoch **134**, 50/214 steps; ViT is still stopped at epoch 1. The frozen-probe and matched-seed queues are running but gated on full seed-0 pretraining. No downstream test scores have been accessed; local result files remain **0/8**.


At **05:48 MDT**, Mamba seed 0 was at epoch **134**, 70/214 steps. ViT remains stopped after epoch 1 pending its guarded handoff, and the evaluation/seed queues remain gated. No downstream test partitions have been evaluated; local scores remain **0/8**.


At **06:01 MDT**, Mamba seed 0 completed epoch **134**; best fixed-validation checkpoint remains epoch **132** (**0.296659710**). ViT remains at **1/240** and is queued behind Mamba. Seed-0 frozen-probe and matched-seed evaluation queues remain gated; no downstream test scores have been read and local results remain **0/8**. The refreshed task-status plot still reports not evaluated.


At **06:20 MDT**, Mamba seed 0 completed epoch **135**; the selected fixed-validation checkpoint remains epoch **132** (**0.296659710**). ViT remains at **1/240**, queued behind Mamba. Seed-0 frozen-probe and matched-seed evaluation queues are gated; no local downstream results exist (**0/8**) and no task test scores have been accessed. The refreshed task-status plot continues to show not evaluated.


At **06:23 MDT**, Mamba seed 0 was in epoch **136**, step 40/214; ViT remains stopped after epoch 1 pending the guarded sequential handoff. Frozen-probe and matched-seed queues remain gated; local downstream results are **0/8**, and test partitions have not been evaluated.


At **06:24 MDT**, Mamba seed 0 was active in epoch **136**, 60/214 steps. ViT remains stopped after epoch 1 pending its guarded handoff; frozen-probe and matched-seed queues remain gated. Local downstream scores remain **0/8**; held-out test partitions have not been evaluated.


At **06:26 MDT**, Mamba seed 0 was active in epoch **136**, 80/214 steps; ViT remains stopped at epoch 1 pending its sequential handoff. Probe, seed, and fine-tuning queues remain gated. Local downstream results remain **0/8**, and held-out test splits have not been evaluated.


At **06:27 MDT**, Mamba seed 0 was active in epoch **136**, 90/214 steps; ViT remains stopped after epoch 1 pending the sequential handoff. Frozen-probe and matched-seed queues remain gated. Local downstream results remain **0/8**, and held-out test splits have not been evaluated.


## Latest run and evaluation gate — 2026-10-09 08:58 MDT

Mamba-2 seed 0 is still active (PID **1941478**) in epoch **144/240**, global step **30,710/51,360** (**108/214** optimizer steps), with rolling mean train JEPA loss **0.251114444**. Its latest completed fixed-protocol validation loss is **0.293338094** (epoch 143); the selected best remains epoch **141**, loss **0.293175797**. The RTX 5070 Ti reports **6,185/16,303 MiB** used at **16%** utilization; the most recent training record estimates about **27h46m** remaining. Checkpoint metadata confirms 143 completed epochs, seed 0, Mamba-2 architecture, resolved configuration, four-source mix, and fixed validation protocol.

ViT seed 0 remains preserved at **1/240**. The GPU-sequential ViT handoff, seed-0 frozen-probe evaluator, matched-seed runs, and fine-tuning queue are live and waiting behind Mamba. There are currently **0/8** seed-0 frozen-probe JSON results; the downstream split manifests have not been created by the evaluator, so no downstream test split has been opened. Once both seed-0 runs finish, the evaluator uses the same persisted sample-level 80/20 indices for both architectures, with three split seeds and training-only feature standardization. Its fixed probe settings require no downstream validation tuning; the fine-tuning path reserves validation indices from training data and reports test scores only after validation selection. The four HDF5 task files are present and read-only. The configured pretraining mixture remains spectrogram, RFP, WiFi CSI, and NRPos Indoor; `5g-csi.h5` remains excluded.


## Epoch 144 checkpoint and current gate — 2026-10-09 09:08 MDT

Verified the Mamba seed-0 epoch-144 completion in `train_console.log`, `run_metadata.json`, and `latest.pt`: global step **30,816**, mean training loss **0.249969702**, fixed validation loss **0.294683115**, and selected best still epoch **141** (**0.293175797**). The run continues in epoch **145** at step **30,830/51,360** (**14/214**), train loss **0.249739387**, ETA about **27h37m**. ViT seed 0 remains at **1/240**. Its handoff, the frozen-probe evaluation, matched-seed queue, and supervised fine-tuning queue remain live and sequentially gated; no downstream results exist, and no test split has been opened. The epoch-144 PNG snapshot and file descriptions are current in `../runs/latent-wave/comparison/current/figure_index.md`.


## Epoch 145 checkpoint and evaluation gate — 2026-10-09 09:27 MDT

The Mamba seed-0 epoch-145 checkpoint is now the best validation checkpoint: validation JEPA loss **0.292463966**, versus **0.293175797** at epoch 141. `best.pt`, `latest.pt`, and run metadata confirm architecture Mamba-2, seed 0, 145 completed epochs, step **31,030**, the fixed validation protocol, and resume state. The process continues in epoch **146**, step **31,050/51,360** (**20/214**). ViT remains at **1/240**. The GPU-sequential ViT handoff, seed-0 probe evaluator, matched-seed queue, and fine-tuning queue remain alive and gated; there are still **0/8** frozen-probe results and no downstream test split has been opened. The updated figure index and PNGs reflect completed training through epoch 145.


## Epoch 146 checkpoint and evaluation gate — 2026-10-09 09:45 MDT

Mamba seed 0 completed epoch **146/240** at step **31,244**, with fixed-validation loss **0.287340588**, now the best checkpoint (previous best epoch 145: **0.292463966**). `best.pt`, `latest.pt`, and metadata confirm the architecture, seed, epoch, step, validation protocol, and resume state. Mamba continues in epoch **147**, step **31,250/51,360** (**6/214**). ViT remains at **1/240**. The sequential ViT handoff, frozen-probe evaluator, matched-seed queue, and fine-tuning queue remain alive and waiting. Frozen-probe outputs remain **0/8**; the downstream split manifests are still uncreated, so no downstream test split has been opened. Refreshed plots and figure index cover completed Mamba epoch 146.


## Epoch 147 checkpoint and evaluation gate — 2026-10-09 10:04 MDT

Mamba seed 0 completed epoch **147/240** at global step **31,458**, with mean train JEPA loss **0.246896229** and fixed-validation loss **0.285505922**, a new best (**0.64%** below epoch 146). The run metadata reports Mamba-2, seed 0, 147 completed epochs, and the fixed validation protocol; `latest.pt` and `best.pt` were updated at 10:02 MDT. The trainer continues in epoch **148**, step **31,470/51,360** (**12/214**), rolling train loss **0.247251548**, ETA about **26h47m**. ViT remains at **1/240**, with its guarded resume waiting. The frozen-probe queue, matched-seed queue, and fine-tuning queue remain alive and gated. There are **0/8** seed-0 probe outputs, and the evaluator has not created split manifests or opened any downstream test split. Refreshed plots and the figure index cover completed training through Mamba epoch 147.

At **10:06 MDT**, Mamba remained active in epoch **148**, step **31,510/51,360** (**52/214**), rolling train loss **0.245781113**, latest completed validation **0.285505922**, ETA about **26h44m**, and GPU use **6,899/16,303 MiB**. The seed-0 downstream result count remains **0/8**; the saved plots remain through completed epoch 147.

At **10:22 MDT**, Mamba seed 0 had completed epoch **148/240** at global step **31,672**, with train loss **0.247056942** and fixed-validation loss **0.285276087**, a new best. The process continues in epoch **149**, step **31,680/51,360** (**8/214**), rolling train loss **0.244748445**, ETA about **26h31m**. The refreshed plots cover completed Mamba epoch 148. ViT remains at **1/240**; all queues remain gated, probe outputs remain **0/8**, and downstream test splits have not been opened.

At **10:23 MDT**, Mamba remained active at epoch **149**, step **31,690/51,360** (**18/214**), rolling train loss **0.248790540**, latest validation **0.285276087**, and ETA about **26h30m**. The seed-0 downstream result count remains **0/8**.
