# LatentWave Mamba-2 and ViT training results

Report updated: 2026-10-09 03:27 MDT. Values are taken from current run metadata, console/JSONL logs, checkpoints, and evaluator output.

## Current verified status — 2026-10-09 03:27 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **127/240**, global step **26,980/51,360** (**16/214** steps in this epoch). Rolling train loss is **0.270056016**; latest completed fixed-validation loss is **0.311929053** from epoch 126. The selected best remains epoch **124**, **0.303736491**. LR is **4.99985e-4**, EMA momentum **0.998101**, elapsed time **129,323 s**, and ETA **116,860 s (~32h28m)**. PyTorch allocation/reservation is **2.23/4.94 GiB**; the RTX 5070 Ti reports **6,813/16,303 MiB** at **15%** utilization. Metadata and `latest.pt` confirm **126 completed epochs/global step 26,964**.

Epoch 126 completed with mean train JEPA loss **0.269352273** and fixed-validation loss **0.311929053**. Relative to epoch 125, training rose **0.65%** and validation rose **1.29%**, so epoch 124 remains best. Per-source validation changed **+2.21%** on spectrogram (**0.243222915**), **+6.12%** on RFP (**0.140455212**), **+0.99%** on WiFi CSI (**0.450498933**), and **−0.45%** on NRPos Indoor (**0.413539153**). Each source contributed **13,696 examples**. Epoch duration was **1,117.082 s** at **49.042 source examples/s**. This is a mixed, higher-validation-loss epoch; no downstream performance conclusion is available. Over completed fixed-validation epochs 20–126, an ordinary least-squares fit has slope **−0.000311 loss/epoch**; over epochs 117–126 it is **−0.000278/epoch**, while the last five epochs 122–126 slope upward at **+0.001444/epoch**. The selected best at epoch 124 (**0.303736491**) is **13.23% below** epoch 20 (**0.350028966**). The measured curve therefore shows a long-run decrease with a recent rebound; whether that rebound persists remains uncertain. No implementation defect has been confirmed, and no downstream metric has informed checkpoint selection.

Refreshed training/validation/source/compute/paper-reference PNGs and machine-readable summaries at **03:27 MDT** through completed epoch **126**, excluding active epoch 127. ViT seed 0 remains at epoch 1; downstream evaluations remain gated and result files remain **0/8**.

## Prior status snapshot — 2026-10-09 03:08 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **126/240**, global step **26,760/51,360** (**10/214** steps in this epoch). Rolling train loss is **0.268116807**; latest completed fixed-validation loss is **0.307943307** from epoch 125. The selected best remains epoch **124**, **0.303736491**. LR is **5.07061e-4**, EMA momentum **0.998084**, elapsed time **128,175 s**, and ETA **117,829 s (~32h44m)**. PyTorch allocation/reservation is **2.23/4.94 GiB**; the RTX 5070 Ti reports **6,813/16,303 MiB** at **16%** utilization. Metadata and `latest.pt` confirm **125 completed epochs/global step 26,750**.

Epoch 125 completed with mean train JEPA loss **0.267618363** and fixed-validation loss **0.307943307**. Relative to epoch 124, training fell **0.28%**, while validation rose **1.39%**, so epoch 124 remains selected. Per-source validation changed **+1.93%** on spectrogram (**0.237965537**), **+10.22%** on RFP (**0.132353933**), **−0.07%** on WiFi CSI (**0.446064762**), and **+0.09%** on NRPos Indoor (**0.415388997**). Each source contributed **13,696 examples**. Epoch duration was **1,102.337 s** at **49.698 source examples/s**. This is a mixed label-free validation epoch; no downstream performance conclusion is available.

Refreshed training/validation/source/compute/paper-reference PNGs and machine-readable summaries at **03:08 MDT** through completed epoch **125**, excluding active epoch 126. ViT seed 0 remains at epoch 1; its sequential continuation and downstream evaluations remain gated on completed seed-0 pretraining. Downstream results are still **0/8**.

## Prior status snapshot — 2026-10-09 02:50 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **125/240**, global step **26,550/51,360** (**14/214** steps in this epoch). Rolling train loss is **0.268343421**; latest completed fixed-validation loss is **0.303736491** from epoch 124 and is the new best. LR is **5.13813e-4**, EMA momentum **0.998068**, elapsed time **127,092 s**, and ETA **118,763 s (~33h00m)**. PyTorch allocation/reservation is **2.23/4.94 GiB**; the RTX 5070 Ti reports **6,813/16,303 MiB** at **16%** utilization. Metadata and `latest.pt` confirm **124 completed epochs/global step 26,536**; `best.pt` selects epoch 124.

Epoch 124 completed with mean train JEPA loss **0.268382410** and fixed-validation loss **0.303736491**. Relative to epoch 123, training fell **0.67%** and validation fell **1.13%**; validation is **0.44%** below the previous best at epoch 122 (**0.305077690**). Per-source validation changed by **−1.10%** on spectrogram (**0.233458334**), **−7.80%** on RFP (**0.120076881**), **+0.17%** on WiFi CSI (**0.446385168**), and **−0.45%** on NRPos Indoor (**0.415025579**). Each source contributed **13,696 examples**. Epoch duration was **1,121.712 s** at **48.840 source examples/s**. This is a fixed, label-free validation improvement; there are still no downstream scores.

Refreshed training/validation/source/compute/paper-reference PNGs and machine-readable summaries at **02:50 MDT** through completed epoch **124**, excluding active epoch 125. A transient process `D` state at the end of epoch 124 cleared; the same trainer completed the epoch and checkpointed normally. ViT seed 0 remains at epoch 1, and downstream task results remain **0/8**.

## Prior status snapshot — 2026-10-09 02:31 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **124/240**, global step **26,330/51,360** (**8/214** steps in this epoch). Rolling train loss is **0.266369982**; latest completed fixed-validation loss is **0.307202494** from epoch 123. The selected best remains epoch **122**, **0.305077690**. LR is **5.20885e-4**, EMA momentum **0.998051**, elapsed time **125,938 s**, and ETA **119,720 s (~33h15m)**. PyTorch allocation/reservation is **2.09/4.04 GiB**; the RTX 5070 Ti reports **5,888/16,303 MiB** at **16%** utilization. Metadata and `latest.pt` confirm **123 completed epochs/global step 26,322**.

Epoch 123 completed with mean train JEPA loss **0.270191717** and fixed-validation loss **0.307202494**. Relative to epoch 122, training loss fell **0.25%** while validation rose **0.70%**, so epoch 122 remains the label-free best. Per-source validation changed by **−0.71%** on spectrogram (**0.236055199**), **+11.01%** on RFP (**0.130228439**), **−0.22%** on WiFi CSI (**0.445642162**), and **−0.41%** on NRPos Indoor (**0.416884176**). Each source contributed **13,696 examples**. Epoch duration was **1,104.463 s** at **49.602 source examples/s**. This is a mixed pretraining validation result; no downstream performance conclusion is available.

Refreshed the loss/compute/paper-reference PNGs and machine-readable summaries at **02:30 MDT** through completed epoch **123**, excluding active epoch 124. The validation plots separate legacy and fixed protocols. ViT seed 0 remains at its preserved epoch-1 checkpoint, the ViT handoff waits for Mamba completion, and downstream results remain **0/8**.

## Prior status snapshot — 2026-10-09 02:13 MDT

Mamba-2 seed 0 PID **1870291** is active at epoch **123/240**, global step **26,120/51,360** (**12/214** steps in the current epoch). Rolling train loss is **0.270566946**; latest completed fixed-validation loss is **0.305077690** from epoch 122, now the best checkpoint. LR is **5.27631e-4**, EMA momentum **0.998034**, elapsed time **124,853 s**, and ETA **120,646 s (~33h31m)**. PyTorch allocation/reservation is **2.09/3.39 GiB**; the RTX 5070 Ti reports **5,888/16,303 MiB** at **16%** utilization. Metadata and `latest.pt` confirm **122 completed epochs/global step 26,108**; `best.pt` selects epoch 122.

Epoch 122 completed with mean train JEPA loss **0.270876031** and fixed-validation loss **0.305077690**. Relative to epoch 121, train loss rose **0.28%** while validation fell **2.21%**, establishing a new best versus epoch 115 (**0.308417416**). Per-source fixed-validation losses were spectrogram **0.237739586**, RFP **0.117315757**, WiFi CSI **0.446641195**, and NRPos Indoor **0.418614221**; these fell **4.28%**, **9.85%**, **0.59%**, and **0.34%** from epoch 121. Each source contributed **13,696 examples**. Epoch duration was **1,122.683 s** at **48.797 source examples/s**. This is validation-only pretraining evidence; no downstream result is available.

Refreshed PNGs and CSV/JSON summaries at **02:12 MDT** now include completed Mamba epoch **122** and exclude in-progress epoch 123. The fixed validation curve is labeled separately from legacy validation. ViT seed 0 remains at its preserved epoch-1 checkpoint; the sequential handoff waits for Mamba to finish 240 epochs. Downstream results remain **0/8**.

## Prior status snapshot — 2026-10-09 00:57 MDT

Mamba-2 seed 0 PID **1717366** is active in epoch **119/240**, global step **25,260/51,360** (**8/214** in this epoch). Rolling mean train loss is **0.274069**; latest completed fixed-validation loss is **0.311163162** from epoch 118. Best checkpoint remains epoch **115** at **0.308417416**. LR is **5.5519e-4**, EMA momentum **0.997967**, elapsed time **120,386 s**, and ETA **124,389 s (~34h33m)**. PyTorch allocation/reservation is **2.22/4.20 GiB**; the latest RTX 5070 Ti poll reported **6,058/16,303 MiB** at **17%** utilization. Metadata and `latest.pt` confirm **118 completed epochs/global step 25,252**; `best.pt` still selects epoch 115.

ViT seed 0 remains at its preserved epoch-1 checkpoint; its console log records a deliberate checkpointed stop after epoch 1, not a training crash. The guarded ViT handoff, seed-0 evaluation, matched-seed, and supervised fine-tuning queues remain alive and gated; only Mamba is using the GPU. Downstream results remain **0/8**; no downstream test score has been used. Figures are current through completed Mamba epoch 118; epoch 119 is in progress.

Epoch 116 completed at global step **24,824** with mean training loss **0.275183419** and fixed-validation loss **0.311378317**. Train loss fell **1.15%** and validation rose **0.96%** from epoch 115, so best remains epoch **115** (**0.308417416**). Per-source validation changed by **+2.79%** on spectrogram (**0.233306512**), **+4.70%** on RFP (**0.123771424**), **+0.40%** on WiFi CSI (**0.459936959**), and **−0.44%** on NRPos Indoor (**0.428498373**) from epoch 115; each source contributed **13,696 examples**. Epoch duration was **1,012.167 s** at **54.125 source examples/s**. From fixed-validation epoch 20, training and validation are **20.11%** and **11.04% lower**, respectively. The one-epoch validation rebound is small relative to the lowest observed checkpoint, but does not change selection. These are label-free pretraining measurements and do not imply a downstream win.

Epoch 117 completed at global step **25,038** with mean training JEPA loss **0.275101022** and fixed-validation loss **0.310250566**. Training fell **0.03%** and validation fell **0.36%** from epoch 116; the selected best remains epoch **115** (**0.308417416**). Per-source validation changed **+1.16%** on spectrogram (**0.236017771**), **−5.13%** on RFP (**0.117427866**), **+0.36%** on WiFi CSI (**0.461598498**), and **−0.59%** on NRPos Indoor (**0.425958127**) from epoch 116. Each source contributed **13,696 examples**. Epoch duration was **1,037.539 s** at **52.802 source examples/s**. Relative to fixed-validation epoch 20, aggregate training and validation losses are **20.13%** and **11.36% lower**. The latest validation improvement is measured pretraining loss only, not evidence of downstream benefit.

Epoch 118 completed at global step **25,252** with mean training JEPA loss **0.277608566** and fixed-validation loss **0.311163162**. Train loss rose **0.91%** and validation rose **0.29%** from epoch 117; best remains epoch **115** (**0.308417416**). Per-source validation changed **+0.89%** on spectrogram (**0.238116181**), **+2.15%** on RFP (**0.119948700**), **−0.07%** on WiFi CSI (**0.461280465**), and **−0.15%** on NRPos Indoor (**0.425307301**) from epoch 117. Each source contributed **13,696 examples**. Epoch duration was **1,073.035 s** at **51.055 source examples/s**. From fixed-validation epoch 20, aggregate training and validation losses remain **19.41%** and **11.10% lower**. This epoch regressed slightly from epoch 117 and does not change checkpoint selection.

The saved Mamba and ViT resolved configurations were compared field by field. They match except for `encoder.architecture` and the architecture-specific `checkpoint.dir`; their dataset paths/keys, transforms, masks, predictor, optimizer and schedule, accumulation/effective batch, and epoch count match. Both run metadata records seed 0. The measured models have different sizes (Mamba **19.09M** total parameters; ViT **13.75M**), so this isolates the encoder choice but is not parameter-count matched.

The fixed validation implementation resets the validation collator and CPU/CUDA RNG for each source using `run_seed + 10,000 + source_index`, and validation loaders use zero workers. Because the resumed ViT has the same seed and source order and its handoff passes `--reset-validation-selection`, its future epochs will use the same masks/channel subsets as Mamba's fixed-protocol segment. The old ViT epoch 1 and Mamba epochs 1–19 remain legacy-protocol points and are not compared directly.

### Throughput sampling

A read-only 30-second sample around **23:18 MDT** found all 16 data-loader workers in sleep state in 6/7 samples; one worker briefly entered `p9_client_rpc` uninterruptible wait in the earlier snapshot, but no worker remained blocked during the sample. The trainer used about **1.07 CPU cores** while GPU utilization varied **15–21%** at roughly **6.05 GiB** device memory. This does not support a sustained HDF5 storage stall; it points to serial CPU-side work or small-batch/launch overhead as possibilities, not a diagnosed cause. No profiler or training change was applied while the run is active.

Epoch 112 completed at global step **23,968** with mean training loss **0.280693547** and fixed-validation loss **0.309308745**. Relative to epoch 111, train loss fell **1.21%** and validation fell **4.39%**. It is a new best, **1.23%** below epoch 108 (**0.313163118**). Per-source validation changed **−9.20%** for spectrogram (**0.227943239**), **−18.33%** for RFP (**0.118405447**), **−0.46%** for WiFi CSI (**0.456578096**), and **−1.14%** for NRPos Indoor (**0.434308199**) versus epoch 111; each source contributed **13,696 examples**. Epoch duration was **1,030.794 s** at **53.147 source examples/s**. From fixed-validation epoch 20, aggregate training loss is **18.51% lower** and validation loss **11.63% lower**. This is label-free pretraining evidence only.

Epoch 113 completed at global step **24,182** with mean training loss **0.278406711** and fixed-validation loss **0.314012933**. Training loss fell **0.81%** from epoch 112 while validation rose **1.52%**, so epoch 112 remains the selected best. Per-source validation increased from epoch 112 by **0.34%** on spectrogram (**0.228721433**), **5.24%** on RFP (**0.124616233**), **1.30%** on WiFi CSI (**0.462518405**), and **1.36%** on NRPos Indoor (**0.440195662**); all sources contributed **13,696 examples**. Epoch duration was **1,023.843 s** at **53.508 source examples/s**. Aggregate validation remains **10.29% below** fixed-validation epoch 20. This is pretraining-only evidence and not a downstream result.

Epoch 114 completed at global step **24,396** with mean training loss **0.280602803** and fixed-validation loss **0.317852290**. Training loss rose **0.79%** and validation rose **1.22%** from epoch 113; best remains epoch **112** at **0.309308745**. Per-source validation changed by **+2.07%** on spectrogram (**0.233463846**), **+9.35%** on RFP (**0.136268295**), **+1.36%** on WiFi CSI (**0.468817742**), and **−1.67%** on NRPos Indoor (**0.432859278**) from epoch 113. Epoch duration was **1,052.895 s** at **52.032 source examples/s**. Aggregate validation is still **9.19% below** fixed-validation epoch 20, though it has regressed over the latest two epochs. This is label-free pretraining evidence only.

The epoch-20 to epoch-112 source breakdown shows nonuniform generalization: training loss fell **12.95%** on spectrogram, **15.51%** on RFP, **17.86%** on WiFi CSI, and **21.76%** on NRPos Indoor. Validation fell **10.62%**, **11.35%**, and **22.70%** on RFP, WiFi CSI, and NRPos Indoor, but rose **19.49%** on spectrogram. This points to a source-specific validation gap (also widening for WiFi CSI), while the equally weighted aggregate still improves. The prior representation audit found no collapse at its sampled scale, and code review found no confirmed mask/alignment/EMA defect. There is no evidence to justify changing the active run; any model/optimizer candidate comparison remains a later validation-only experiment after the matched baseline.

Epoch 111 completed at global step **23,754** with mean training loss **0.284132551** and fixed-validation loss **0.323513525**. Relative to epoch 110, train loss rose **1.33%** and validation rose **2.90%**. Best remains epoch **108** at **0.313163118**; epoch 111 is **3.31%** above the selected checkpoint. Per-source validation was spectrogram **0.251045576** (**+9.79%**), RFP **0.144985312** (**+13.06%**), WiFi CSI **0.458695542** (**−0.88%**), and NRPos Indoor **0.439327670** (**+0.31%**) versus epoch 110; each source contributed **13,696 examples**. Epoch duration was **1,016.515 s** at **53.894 source examples/s**. Relative to fixed-validation epoch 20, mean training loss is **17.51% lower** and aggregate validation loss is **7.58% lower**; however, epoch 111 reverses part of the recent validation improvement. This is label-free pretraining evidence only.

Epoch 110 completed at global step **23,540** with mean training loss **0.280395926**, fixed-validation loss **0.314407935**, duration **1,029.625 s**, and throughput **53.208 source examples/s**. Relative to epoch 109, train loss fell **0.0047%** and validation rose **0.3124%**. Epoch 108 remains the selected best at **0.313163118**; epoch 110 is **0.3975%** above it. Source validation losses were spectrogram **0.228660619**, RFP **0.128242104**, WiFi CSI **0.462778020**, and NRPos Indoor **0.437950996**; changes from epoch 109 were **−1.66%**, **+10.61%**, **−0.57%**, and **−0.43%**, respectively. Aggregate validation remains **10.18%** below fixed-validation epoch 20 (**0.350028966**). This remains pretraining-only evidence.

Epoch 109 completed at global step **23,326** with mean training loss **0.280409035**, fixed-validation loss **0.313428790**, duration **1,022.554 s**, and throughput **53.576 source examples/s**. Relative to epoch 108, train loss fell **1.63%** and validation rose **0.0848%**. Epoch 108 remains the selected best at **0.313163118**. Source validation losses were spectrogram **0.232509777**, RFP **0.115942584**, WiFi CSI **0.465417056**, and NRPos Indoor **0.439845744**; changes from epoch 108 were **+0.83%**, **−3.12%**, **+1.15%**, and **−0.54%**, respectively. Aggregate validation remains **10.46%** below epoch 20 (**0.350028966**). This is pretraining-only evidence.

Epoch 108 completed at global step **23,112** with mean training loss **0.285069486**, fixed-validation loss **0.313163118**, duration **1,031.099 s**, and throughput **53.132 source examples/s**. Relative to epoch 107, training loss fell **1.98%** and validation fell **2.77%**. Epoch 108 is **0.17%** below the prior best at epoch 104 (**0.313698007**) and is now the selected checkpoint. Per-source validation losses were spectrogram **0.230604566**, RFP **0.119670777**, WiFi CSI **0.460145604**, and NRPos Indoor **0.442231526**; each source contributed **13,696 examples**. All four fell from epoch 107 by **4.79%**, **11.07%**, **1.52%**, and **0.47%**, respectively. Aggregate validation is **10.53%** below fixed-validation epoch 20 (**0.350028966**). This remains pretraining-only evidence and does not establish downstream results.

Epoch 107 completed at global step **22,898** with mean train loss **0.290839685**, fixed-validation loss **0.322082475**, duration **1,033.740 s**, and throughput **52.996 source examples/s**. Relative to epoch 106, training loss rose **1.43%** and validation loss rose **1.97%**. The best checkpoint remains epoch **104** at **0.313698007**; epoch 107 is **2.67%** above it. Source validation losses were spectrogram **0.242199295**, RFP **0.134562108**, WiFi CSI **0.467261661**, and NRPos Indoor **0.444306836**; each contributed **13,696 examples**. Changes from epoch 106 were **+5.49%**, **+5.82%**, **+1.19%**, and **−0.14%**, respectively. Validation remains **7.98%** below epoch 20 (**0.350028966**), but epoch 107 reversed part of the preceding improvement. This is pretraining-only evidence and does not establish downstream performance.

Epoch 106 completed at global step **22,684** with mean train loss **0.286735468**, fixed-validation loss **0.315858892**, duration **1,035.348 s**, and throughput **52.914 source examples/s**. Validation improved **0.189%** from epoch 105, but the selected best remains epoch **104** at **0.313698007** (epoch 106 is **0.689%** above it). Source validation losses were spectrogram **0.229599750**, RFP **0.127155985**, WiFi CSI **0.461755463**, and NRPos Indoor **0.444924368**; each source contributed **13,696 examples**. The aggregate fixed-validation loss is **9.76%** lower than epoch 20 (**0.350028966**). Source movement from epoch 105 is mixed: spectrogram improved **2.01%**, while RFP, WiFi CSI, and NRPos Indoor rose **0.87%**, **0.19%**, and **0.07%**. This is label-free pretraining evidence, not a downstream result.

Epoch 105 completed at global step **22,470** with mean train loss **0.286679862**, fixed-validation loss **0.316456184**, duration **1,048.179 s**, and throughput **52.266 source examples/s**. Validation rose **0.879%** from epoch 104, so the best checkpoint remains epoch **104** at **0.313698007**. Epoch 105 source validation losses were spectrogram **0.234302539**, RFP **0.126058910**, WiFi CSI **0.460858669**, and NRPos Indoor **0.444604617**; every source contributed **13,696 examples**. From fixed-validation epoch 20 to epoch 105, aggregate training loss fell **16.77%** and validation loss fell **9.59%**. Between epochs 104 and 105, validation increased **3.95%** on spectrogram, **3.37%** on RFP, and **0.03%** on WiFi CSI, while NRPos Indoor decreased **0.47%**. The noisy source-specific movement and aggregate improvement do not identify a confirmed implementation defect or imply downstream gains.

The fixed-validation log-only trend check now uses **93 completed epochs (20–112)**. Aggregate validation’s ordinary least-squares slopes are **−0.000285 per epoch** over all records, **−0.000513 per epoch** over epochs 93–112, and **−0.000325 per epoch** over epochs 103–112. Across epochs 20–112, source validation slopes are spectrogram **+0.000432**, RFP **−0.000034**, WiFi CSI **−0.000273**, and NRPos Indoor **−0.001272** per epoch. Source train/validation gap slopes are spectrogram **+0.000549/epoch** (latest gap **0.053938**), WiFi CSI **+0.000451/epoch** (latest gap **0.073114**), RFP **+0.000062/epoch** (latest gap **−0.002816**), and NRPos Indoor **−0.000015/epoch** (latest gap **−0.009775**). Epoch 112 improved aggregate and all four source validation losses; the cumulative source-specific gaps remain uneven. These unlabeled trends motivate validation-based diagnosis but do not establish downstream improvement or a code defect.

`latest.pt` and run metadata last updated at **22:32 MDT**. Metadata confirms **110 completed epochs**, global step **23,540**, last validation **0.314407935**, and best validation **0.313163118** from epoch **108**. The guarded ViT handoff, evaluation gate, matched-seed queue, and fine-tuning queue remain alive and waiting; none starts another GPU run while Mamba is active. ViT seed 0 remains at its preserved epoch-1 checkpoint. No downstream result JSON exists (**0/8**) and the shared split directory is absent.

The training-history, source-loss, validation-protocol, task-status, Table II, and compute PNGs were refreshed at **23:26 MDT** through completed Mamba epoch 113 and ViT epoch 1. Duplicate copied/resumed snapshots are collapsed in the compute figure; its machine-readable rows are in `../runs/latent-wave/comparison/current/pretraining_compute_snapshots.csv`. All four local downstream tasks remain not run. Table II values remain labeled as paper-reported references and Table III remains reference-only. Full pretraining and downstream evaluation remain incomplete. No training process, checkpoint, config, or HDF5 file was changed by the plot refresh.

## Prior status snapshot — 2026-10-08 18:58 MDT

Mamba-2 seed 0 is running on the RTX 5070 Ti. The original 19-epoch legacy-validation segment remains preserved; the fixed-validation continuation has completed epochs 20–98. At the 18:55 MDT poll, PID 1717366 was in epoch 99/240 at global step 20,990/51,360 (18/214 in this epoch), with rolling train loss 0.291422, latest fixed-validation loss 0.318487 (epoch 98), LR 6.8793e-4, EMA 0.997635, cumulative elapsed 99,733 s (~27h42m), and ETA 144,301 s (~40h05m). PyTorch memory was 1.97 GiB allocated / 2.37 GiB reserved; the RTX 5070 Ti reported 4,153/16,303 MiB at 22% utilization. The completed latest and selected best checkpoints are both epoch 98/global step 20,972; the selected JEPA validation loss is 0.318486981. From fixed-validation epoch 20, mean train loss fell 15.73% and validation fell 9.01%. These are pretraining-loss results, not downstream scores.

ViT seed 0 remains at its preserved epoch-1 checkpoint. Its console log records a handled SIGTERM during epoch 1, followed by the trainer finishing the epoch and saving the checkpoint normally; there is no traceback. The log does not identify who sent the signal or why. PID 1182826 waits for Mamba completion before resuming ViT under fixed validation. The seed-0 evaluator (PID 1186826), matched-seed queue (PID 1470912), and fine-tuning queue (PID 1471062) remain gated; downstream results remain 0/8. HDF5 inputs are read-only.

### Stop, diagnosis, and resume — 2026-10-08 18:58 MDT

Following the user's request, SIGTERM was sent only to Mamba trainer PID 1149902. Its handler finished epoch 98, evaluated, saved latest.pt and best.pt, recorded stopped_after_epoch, and exited. Epoch 98 fixed validation was 0.318486981, a new best versus epoch 90 (0.320326810). CPU inspection confirmed Mamba-2/seed 0, 98 completed epochs/global step 20,972, 19,085,440 finite model parameters, and optimizer, scheduler, sampler, RNG, resolved-config, and revision state.

A read-only target-encoder representation audit ran on 64 evenly spaced training and 64 validation examples per source, without labels. Mean-pooled features had nonzero variance in every embedding dimension; the fraction of dimensions with standard deviation below 1e-4 was zero for all four sources. Pooled covariance effective ranks ranged 8.73–15.53, with maximum observable rank 63 for 64 audited samples. This gives no evidence of representation collapse at this sample scale. The machine-readable audit is ../runs/latent-wave/diagnostics/mamba2_seed0_epoch98_representation.json.

From fixed-validation epoch 20 to 98, source validation changed +14.15% for spectrogram (0.190764→0.217764), +2.73% for RFP (0.132477→0.136096), −9.13% for WiFi CSI (0.515054→0.468020), and −19.54% for NRPos Indoor (0.561820→0.452068); each source contributes 13,696 examples per epoch. Code review found no confirmed loss alignment, masking, source weighting, target freeze/EMA, or validation bug. The data show aggregate learning with a noisy late plateau and source-specific generalization drift. No hyperparameter or architecture change was justified by this diagnostic alone.

The 240-epoch prose preset was resumed in the foreground from the epoch-98 checkpoint, preserving optimizer/scheduler and validation selection. The restart omitted the validation-selection reset because the fixed protocol was unchanged:

~~~bash
set -o pipefail
./.venv/bin/python -u src/pretrain.py \
  --config configs/mamba2_prose.yml \
  --device cuda \
  --seed 0 \
  --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
  --resume ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/latest.pt \
  2>&1 | tee -a ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/train_console.log
~~~

The CLI was rechecked; it supports config, device, seed, output directory, and resume. Preflight returned shape (48, 28, 256) on cuda:0. No code, config, data, or optimizer setting changed. At the 18:55 MDT poll, the resumed process was at epoch 99/240, step 20,990/51,360 (18/214), rolling train loss 0.291422, latest validation 0.318487, LR 6.8793e-4, EMA 0.997635, ETA 144,301 s (~40h05m), PyTorch memory 1.97/2.37 GiB, and RTX 5070 Ti use 4,153/16,303 MiB at 22%. PNGs and summary tables now include completed epoch 98. Local downstream results remain unavailable.

### Queue recovery and live snapshot — 2026-10-08 19:07 MDT

The user-requested stop caused the guarded ViT handoff, seed-0 evaluation gate, matched-seed queue, and fine-tuning queue to exit after detecting Mamba metadata stopped_after_epoch 98. Their logs confirm no evaluation or additional training started. I preserved the old ViT handoff script as ../runs/latent-wave/full_runs/continue_vit_after_mamba.stale_pid_20261008.bak and fixed the active script to discover the Mamba trainer by its run directory rather than the stale PID from the prior run. Relaunched foreground waiters are alive: ViT handoff PID 1722296, seed-0 evaluation gate PID 1722311, matched-seed queue PID 1722343, and fine-tuning queue PID 1722366. All are waiting, and only Mamba is using CUDA.

At the 19:07 MDT poll, Mamba-2 seed 0 PID 1717366 was in epoch 99/240 at step 21,100/51,360 (128/214), rolling train loss 0.289380, latest fixed validation 0.318487 from epoch 98, LR 6.8465e-4, EMA 0.997643, elapsed 100,232 s, ETA 143,745 s (~39h56m), PyTorch allocated/reserved memory 2.12/3.49 GiB, and GPU use 5,318/16,303 MiB at 19%. Latest complete checkpoint remains epoch 98/global step 20,972; epoch 99 is not complete. The queues are documented in the implementation log.

## Prior status snapshot — 2026-10-08 18:34 MDT

Mamba-2 seed 0 is running on the RTX 5070 Ti. The original 19-epoch legacy-validation segment remains preserved; the fixed-validation continuation has completed epochs 20–97. At **18:34 MDT**, PID **1149902** was in epoch **98/240**, global step **20,770/51,360** (**12/214** in the epoch), with rolling train loss **0.289178**, latest completed fixed-validation loss **0.320756** (epoch 97), LR **6.9447e-4**, EMA **0.997618**, cumulative elapsed **98,663 s (~27h24m)**, and ETA **145,311 s (~40h22m)**. PyTorch memory is **2.42 GiB allocated / 6.14 GiB reserved**; the RTX 5070 Ti reported **7,971/16,303 MiB** at **18%** utilization. The epoch-97 checkpoint records 97 completed epochs at global step **20,758**; its checkpoint `epoch=96` is zero-based and contains optimizer, scheduler, sampler, RNG, resolved-config, and code-revision state. Best validation remains human epoch **90** at **0.320326810**. Epoch 97 mean train loss was **0.290613**, down **1.31%** from epoch 96, and fixed-validation loss fell **0.27%**. Across epochs 20–97, aggregate train and fixed-validation losses are down **15.63%** and **8.36%**; mean validation across epochs 70–97 is **0.325066**. These are pretraining-loss results, not downstream scores.

ViT seed 0 received **SIGTERM** during epoch 1; its handler checkpointed epoch 1 and exited normally, with no traceback. The exact sender/reason is not recorded in the log. Guarded PID **1182826** waits for Mamba completion, then resumes that preserved checkpoint with the fixed validation protocol. The seed-0 evaluator (PID **1186826**), matched-seed queue (PID **1470912**), and fine-tuning queue (PID **1471062**) are still alive and gated; downstream results remain **0/8**. Only Mamba is using the GPU. The comparison summary and PNGs were refreshed at **18:33 MDT** through completed epoch **97**; epoch 98 is in progress. HDF5 inputs reside on the WSL `/mnt/c` 9p mount.

### Implementation review — 2026-10-08 14:27 MDT

Read-only review of the active training path checked patch/mask order, predictor target alignment, source weighting, gradient accumulation, scheduler stepping, target freezing, post-optimizer EMA updates, and the fixed validation RNG reset. The implementation applies the same predictor mask indices to the target tokens, averages the four configured sources equally, uses eight source microbatches per optimizer update (effective aggregate batch 256), and updates the target encoder after the student optimizer step. Config and run-metadata parity checks confirm both seed-0 runs use the same four HDF5 paths/keys, masks, predictor, optimizer/schedule, seed, 214 updates per epoch, 51,360 total updates, and effective batch 256; `5g-csi.h5` is absent. The measured total parameter counts are **19,085,440** for Mamba-2 and **13,752,064** for ViT, a **38.8%** increase from the Mamba-2 architecture. No specific loss, mask, target-EMA, or validation correctness bug was confirmed, so the active run remains unchanged. The fixed-validation aggregate decreased **7.09%** from epoch 20 to epoch 85 and has a minimum at epoch 72; the later fluctuations indicate a plateau rather than no training at all. No downstream scores exist yet, so this does not establish whether Mamba-2 improves task performance.

### Diagnostic review — 2026-10-08 12:10 MDT

The concern that training has stopped improving is not supported by the matched fixed-validation history: mean train loss decreased from **0.344455** at epoch 20 to **0.308285** at epoch 76 (**−10.50%**), and fixed-validation loss decreased from **0.350029** to **0.325567** (**−6.99%**). The minimum remains epoch 72 at **0.320416**; later validation is noisy rather than monotonic. After RFP drove epoch 75's rise, its validation loss fell **0.026282 (15.45%)** at epoch 76; the aggregate validation loss fell **2.02%**. The epoch-76 train loss rose **0.72%** from epoch 75, so this is validation improvement, not a claim of monotonic training improvement.

Read-only inspection found no confirmed loss, EMA, masking, or validation implementation bug. Validation uses the same fixed unlabeled examples and resets its per-source mask/channel RNG; it evaluates every validation example once. The target encoder stays frozen/eval and receives EMA updates only after student optimizer steps. RFP's fixed split is not grossly label-skewed: its 296 validation examples have source labels **76/84/77/59** versus **1,414/1,406/1,413/1,403** in train; these labels are not used by JEPA. This rules out resampled validation and an obvious class-count split failure. Epoch 76's RFP loss reversal shows the epoch-75 spike was not a persistent divergence, though its cause remains unknown. That diagnostic review made no code or config changes; the active baseline continued under its original settings. Use downstream validation results before deciding on a bounded tuning trial.

### Historical fixed-validation source trends through epoch 97

Recomputed from the **78** fixed-protocol epoch records in `../runs/latent-wave/comparison/current/pretraining_source_loss_traces.csv`, aggregate training loss fell **15.63%** from **0.344455** at epoch 20 to **0.290613** at epoch 97; validation fell **8.36%** from **0.350029** to **0.320756**. The best fixed-validation epoch remains **90** at **0.320327**, narrowly below epoch 72 (**0.320416**); the mean validation loss across epochs 70–97 is **0.325066**, consistent with a noisy late plateau. Source validation losses from epochs 20 to 97 changed by **+15.76%** for spectrogram (**0.190764→0.220820**), **+2.84%** for RFP (**0.132477→0.136237**), **−8.53%** for WiFi CSI (**0.515054→0.471142**), and **−19.04%** for NRPos Indoor (**0.561820→0.454826**). Each source contributes the same **13,696 examples** per completed epoch, so these aggregate changes are not explained by logged sample-count imbalance. From epoch 96 to 97, validation fell **0.27%**; the best checkpoint remains epoch 90. These are JEPA validation results, not downstream gains or evidence of a data defect. Keep the baseline objective and source weights unchanged; consider a bounded validation-only tuning trial only after the matched baseline is complete.

### Runtime sample — 2026-10-08 12:27 MDT

A six-second read-only sample (`vmstat 1 6`, `nvidia-smi dmon -s puc -d 1 -c 6`, and worker `ps`) showed **94–98% host CPU idle**, **0% reported I/O wait**, GPU SM utilization **17–20%**, and memory-controller utilization **4–5%**. The trainer used about **106% of one CPU core**; four of the 16 HDF5 loader workers were briefly in `D` state with `p9_client_rpc` wait channels, while other workers slept in the DataLoader poll. All four configured HDF5 inputs are on `/mnt/c`; this makes 9p/HDF5 access latency a plausible contributor to low GPU occupancy, but the snapshot does not prove it is the main bottleneck. The WSL ext4 filesystem has **896 GB** free, enough to stage the roughly 41 GB of inputs, but no copy or I/O-intensive probe was started during active Mamba training. Reassess after the Mamba run and preserve the exact data/config for the matched ViT baseline.

### Runtime recheck — 2026-10-08 16:47 MDT

A second six-second read-only sample during Mamba epoch 92 showed **93–98% host CPU idle**, **0–1% I/O wait**, GPU SM utilization **14–22%**, and memory-controller utilization **3–9%**. The trainer still consumed about **106% of one CPU core**, while all 16 `pt_data_worker` processes slept in `poll_schedule_timeout` with **0.2–0.6% CPU** each. This snapshot did not show workers blocked on 9p or active storage waits. It supports a general CPU-side input/dispatch bottleneck as one possibility, but it does not distinguish Python collation, batch preparation, synchronization, and kernel-launch overhead. No worker count, data location, training process, or checkpoint was changed while the job was active; profile the pipeline after this baseline finishes before adjusting it for subsequent runs.

### Mamba epoch 76 result and refreshed figures — 2026-10-08 12:22 MDT

Mean train JEPA loss was **0.308284640** and fixed-validation loss was **0.325566707**. Train loss rose **0.72%** from epoch 75, while validation fell **2.02%**. The selected best remains epoch 72 (**0.320415764**). Compared with fixed-validation epoch 20 (**0.344455406 train / 0.350028966 validation**), epoch 76 train loss is **10.50% lower** and validation loss **6.99% lower**. Validation changes from epoch 75 were spectrogram **+0.001432 (+0.72%)**, RFP **−0.026282 (−15.45%)**, WiFi CSI **−0.002172 (−0.46%)**, and NRPos Indoor **+0.000215 (+0.04%)**. Epoch duration was **1,033.527 s** at **53.007 source examples/s**. RFP's large epoch-75 increase partially reversed; the source loss alone does not identify the cause.

The checkpoint at `../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/latest.pt` was saved at **12:20:52 MDT** and CPU-audited: Mamba-2/seed 0, zero-based epoch **75** (76 completed), global step **16,264**, fixed-validation protocol, **478 finite tensors / 19,085,440 values**, and optimizer, scheduler, sampler, RNG, resolved-config, and code-revision state. The selected `best.pt` remains epoch **72**. At **12:22 MDT**, the live process had entered epoch **77/240**, global step **16,280/51,360** (**16/214**), rolling train loss **0.305297**, prior fixed validation **0.325567**, LR **8.1765e-4**, EMA **0.997268**, ETA **166,084 s (~46h08m)**, and GPU use **8,034/16,303 MiB**. ViT and downstream queues remain gated; local downstream outputs remain **0/8**.

Refreshed the machine-readable summaries and PNGs through completed Mamba epoch 76 under `../runs/latent-wave/comparison/current/`. Visual inspection confirmed the training and source-loss plots include epoch 76; validation panels separate legacy from fixed protocol; the paper comparison still labels Table II values as paper references and local downstream slots as unevaluated. `figure_index.md` was updated to this snapshot. Commands used from the repository root:

```bash
./.venv/bin/python -u src/summarize_training_runs.py \
  --run-dirs \
    ../runs/latent-wave/full_runs/mamba2_prose_seed0 \
    ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
    ../runs/latent-wave/full_runs/vit_prose_seed0 \
    ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val \
  --out-dir ../runs/latent-wave/comparison/current
./.venv/bin/python -u src/plot_training_history.py \
  --run-dirs \
    ../runs/latent-wave/full_runs/mamba2_prose_seed0 \
    ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
    ../runs/latent-wave/full_runs/vit_prose_seed0 \
    ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val \
  --out-dir ../runs/latent-wave/comparison/current
./.venv/bin/python -u src/plot_compute_comparison.py \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --out-dir ../runs/latent-wave/comparison/current
./.venv/bin/python -u src/plot_paper_comparison.py \
  --paper-reference docs/mamba2_experiment/paper_results_reference.csv \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --downstream-status ../runs/latent-wave/comparison/current/downstream_status.csv \
  --out-dir ../runs/latent-wave/comparison/current
```

No downstream evaluator has run; the four task slots remain marked not run.

Epoch 66 mean training JEPA loss was **0.314207** and fixed-validation loss was **0.326274**, down **0.003231 (0.98%)** from epoch 65. It remains **0.44% above** the selected best, epoch 50 (**0.324860**). Compared with fixed-protocol epoch 20 (**0.344455 train / 0.350029 validation**), epoch 66 is **8.78% lower in training loss** and **6.79% lower in validation loss**. This indicates pretraining-loss improvement with a noisy later plateau; no downstream result is available to establish a model benefit. The completed epoch-66 checkpoint records Mamba-2/seed 0, zero-based epoch 65, global step 14,124, fixed validation, optimizer/scheduler/sampler/RNG states, and resolved config. `best.pt` remains selected by fixed-validation JEPA loss at epoch 50.

Epoch 67 mean training loss fell to **0.311202**, while fixed-validation loss rose to **0.329447** (**+0.97%** from epoch 66); the best remains epoch 50, and epoch 67 is **1.41% above** it. Since fixed-protocol epoch 20, train loss is **9.65% lower** and validation loss **5.88% lower**. Source validation changes from epoch 66 were mixed: spectrogram **+0.009282**, RFP **+0.011426**, WiFi CSI **−0.006336**, and NRPos Indoor **−0.001680**. The latest aggregate rise was driven by spectrogram and RFP; it does not establish a data defect or downstream effect.

Epoch 68 mean training loss was **0.310252** and fixed-validation loss **0.326872**, down **0.78%** from epoch 67; it remains **0.62% above** the selected epoch-50 best (**0.324860**). From fixed-validation epoch 20, train loss is **9.93% lower** and validation loss **6.62% lower**. Compared with epoch 67, validation improved on spectrogram (**−0.006149**), RFP (**−0.007561**), and NRPos Indoor (**−0.001313**), while WiFi CSI rose (**+0.004725**). This is a measured pretraining-loss change only.

Epoch 69 mean training loss was **0.306402** and fixed-validation loss **0.324896**, down **0.60%** from epoch 68. It is just **0.011% above** the best validation checkpoint from epoch 50 (**0.324860**), so checkpoint selection did not change. Relative to epoch 20 (**0.344455 train / 0.350029 validation**), epoch 69 is **11.05% lower in training loss** and **7.18% lower in validation loss**. Validation improved from epoch 68 on RFP (**−0.005791**), WiFi CSI (**−0.001763**), and NRPos Indoor (**−0.001265**), while spectrogram rose **+0.000917**. Epoch time was **1,024.825 s** at **53.457 source examples/s**. This is evidence of continued pretraining-loss improvement; no downstream result exists yet.

Epoch 70 mean training loss was **0.306554** and fixed-validation loss **0.322346**. Training loss rose **0.05%** from epoch 69, while validation fell **0.78%** and set a new best, **0.77% below** the prior epoch-50 minimum (**0.324860**). Relative to fixed-validation epoch 20, epoch 70 is **11.00% lower in training loss** and **7.91% lower in validation loss**. Validation improved from epoch 69 on RFP (**−0.005554**), WiFi CSI (**−0.001190**), and NRPos Indoor (**−0.003671**), with a small spectrogram increase (**+0.000215**). Epoch time was **1,040.497 s** and throughput **52.652 source examples/s**. This improves the pretraining validation criterion; downstream performance remains unmeasured.

Epoch 71 mean training loss was **0.308306** and fixed-validation loss **0.325024**, up **0.83%** from epoch 70. The selected best at that point remained epoch 70 (**0.322346**). Relative to fixed-validation epoch 20 (**0.344455 train / 0.350029 validation**), epoch 71 train loss was **10.50% lower** and validation loss **7.14% lower**. Validation rose on every source from epoch 70: spectrogram **+0.002593**, RFP **+0.004498**, WiFi CSI **+0.002867**, and NRPos Indoor **+0.000751**. Epoch time was **1,014.415 s** at **54.006 source examples/s**. This was a temporary aggregate loss rise; epoch 72 then improved on all four source validation losses.

Epoch 72 mean training loss was **0.304235** and fixed-validation loss **0.320416**, down **1.32%** and **1.42%** from epoch 71, respectively. It set a new best, **0.60% below** the previous best at epoch 70 (**0.322346**). Relative to fixed-validation epoch 20, epoch 72 train loss is **11.68% lower** and validation loss **8.46% lower**. All four source validation losses fell from epoch 71: spectrogram **−0.000977**, RFP **−0.010217**, WiFi CSI **−0.002893**, and NRPos Indoor **−0.004345**; RFP had the largest single-epoch improvement. Epoch duration was **1,047.926 s**, with throughput **52.278 source examples/s**. This is a measured pretraining validation improvement; downstream performance remains unmeasured. PNGs were regenerated through this completed epoch.

Epoch 73 mean training loss was **0.306399** and fixed-validation loss **0.327584**, up **0.71%** and **2.24%** from epoch 72. The selected best remains epoch 72 (**0.320416**). Relative to fixed-validation epoch 20 (**0.344455 train / 0.350029 validation**), epoch 73 train loss is **11.05% lower** and validation loss **6.41% lower**. Validation rose on all sources from epoch 72: spectrogram **+0.005685**, RFP **+0.013613**, WiFi CSI **+0.007493**, and NRPos Indoor **+0.001882**; RFP had the largest rise. Epoch duration was **1,055.526 s**, with throughput **51.902 source examples/s**. This reverses epoch 72's improvement, while remaining below epoch 20 on the aggregate metrics. It does not establish a downstream effect.

Epoch 74 mean training loss was **0.308082** and fixed-validation loss **0.323733**. Training loss rose **0.55%**, while validation fell **1.18%** from epoch 73. The selected best remains epoch 72 (**0.320416**); epoch 74 is **1.04% above** it. Relative to fixed-validation epoch 20 (**0.344455 train / 0.350029 validation**), epoch 74 train loss is **10.56% lower** and validation loss **7.51% lower**. From epoch 73, validation fell for spectrogram (**−0.004723**), RFP (**−0.009701**), and NRPos Indoor (**−0.002890**), while WiFi CSI rose slightly (**+0.001909**). From epoch 20 to epoch 74, fixed validation is **+2.72%** for spectrogram, **+0.56%** for RFP, **−6.51%** for WiFi CSI, and **−13.81%** for NRPos Indoor. The epoch took **1,053.308 s** at **52.011 source examples/s**. The checkpoint audit confirms the latest checkpoint is finite and resumable; it did not replace the epoch-72 best. This is validation-loss progress, not evidence of downstream improvement.

Epoch 75 mean training loss was **0.306074** and fixed-validation loss **0.332268**. Training improved **0.65%** from epoch 74, while validation worsened **2.64%**. It is **3.70% above** the selected epoch-72 best (**0.320416**). Relative to fixed-validation epoch 20 (**0.344455 train / 0.350029 validation**), epoch 75 train loss is **11.14% lower** and validation loss **5.07% lower**. Per-source validation changes from epoch 74 were spectrogram **+0.002565 (+1.31%)**, RFP **+0.036878 (+27.68%)**, WiFi CSI **−0.004669 (−0.97%)**, and NRPos Indoor **−0.000631 (−0.13%)**. The RFP increase dominates the aggregate regression; its cause is unknown. Epoch time was **1,049.030 s**, throughput **52.223 source examples/s**. CPU inspection found 478 finite model tensors and full resume state in the epoch-75 checkpoint; `best.pt` remains epoch 72. This single validation result does not establish a downstream effect.

Recomputed per-source trends from fixed-validation records for epochs 20–72 show validation loss **+2.22%** for spectrogram, **−2.40%** for RFP, **−8.33%** for WiFi CSI, and **−13.63%** for NRPos Indoor. Corresponding training losses changed **−6.51%**, **−7.93%**, **−13.77%**, and **−12.72%**. Epoch 71 had briefly raised RFP validation loss, followed by its largest single-epoch reduction at epoch 72; the cumulative trend no longer supports a persistent RFP validation regression. Spectrogram remains the only source whose epoch-72 validation loss is above its epoch-20 value. These are descriptive trends, not evidence of a data defect. Keep the matched baseline unchanged; if downstream metrics confirm underperformance, test a single source-related hypothesis using validation only.

Epoch 73 reversed the previous single-epoch improvement: all four fixed-validation source losses rose from epoch 72, led by RFP (**+10.53%**) and spectrogram (**+2.92%**). The fixed validation set is unchanged, so these are model-parameter effects across epochs rather than resampling noise; their cause is not established. Across epochs 20–73, aggregate train and validation losses remain **11.05%** and **6.41%** lower, respectively. Continue the matched baseline without source weighting or other changes.

### Runtime profile — 2026-10-08 08:58 MDT

A read-only 10-second sample measured SM utilization at **14–20%** and GPU memory-controller utilization at **3–7%**. `vmstat` showed **94–95%** host CPU idle and **0%** I/O wait; the trainer process used about **106%** CPU, while each of 16 data workers used at most **0.6%**. Static code inspection found repeated CUDA-to-host synchronization points in the training path: channel grouping calls `torch.unique(...).tolist()` once per source microbatch in `src/models/jepa.py`, and each resulting loss group is synchronously checked for finiteness and converted to a Python float in `src/pretrain_jepa.py`. With **1,712** microbatches per epoch and four sources, the channel-list conversion alone occurs at least **6,848** times per epoch; loss groups add more synchronization. This is a concrete avoidable runtime cost and is consistent with low GPU utilization, but its exact throughput impact has not been profiled and it does not explain the validation plateau. The active process has not been changed; any refactor must preserve the current run and be checked on a later short run before adopting it.

The run metadata confirms **65 completed epochs** at global step **13,910**. The latest completed checkpoint stores Mamba-2/seed 0, zero-based epoch **64**, fixed-validation protocol, and optimizer, scheduler, sampler, resolved-config, and RNG state. The selected `best.pt` is zero-based epoch **49** (epoch 50), with validation loss **0.324860**. Epoch 65 validation (**0.329504**) rose by **0.000581 (0.18%)** from epoch 64 (**0.328923**) and remains **1.43% above** the selected best. The fixed-validation mean over epochs 54–65 is **0.331274**. Relative to epoch 20 (**0.350029**), epoch 65 validation is **5.86% lower**; mean training loss fell **8.64%** over the same interval. The history shows early improvement followed by a noisy plateau, not a demonstrated downstream benefit. Legacy Mamba epochs 1–19 and the single ViT epoch use an unrecorded legacy validation protocol and are not directly comparable to the fixed protocol.

### Source-wise loss diagnosis through epoch 65

The fixed validation protocol makes the recent fluctuations traceable by source. From epoch 59 to 60, validation rose **0.010336** overall; RFP contributed **+0.028564**, spectrogram **+0.008086**, WiFi CSI **+0.003958**, and NRPos Indoor **+0.000738**. At epoch 61, all four fell and aggregate validation returned to **0.329126**. From epoch 61 to 62, aggregate validation rose **0.004375**; WiFi CSI contributed **+0.009560**, RFP **+0.004935**, spectrogram **+0.002959**, and NRPos Indoor **+0.000046**. Thus the fluctuations are not isolated to one source, and their cause remains unknown.

Across fixed-validation epochs 20 to 65, NRPos Indoor train/validation loss fell **9.75% / 11.07%**, WiFi CSI fell **12.49% / 7.72%**, and spectrogram train loss fell **6.49%** while its validation loss rose **2.08%**. RFP train and validation losses rose **5.29% / 11.97%**. From epoch 64 to 65, validation increased by **0.001642** for spectrogram, **0.003424** for RFP, and **0.001039** for WiFi CSI, while NRPos Indoor fell **0.003780**. RFP remains a cumulative source-specific weakness; recent aggregate fluctuations involve multiple sources. The available traces do not establish whether the source trends come from optimization, data characteristics, or expected variation. No source weighting, masks, checkpoint rule, or configuration was changed. The selected checkpoint remains epoch 50 by aggregate validation loss.

Source inspection confirms the validation fix is shared across the two configs: identical seed and source ordering give the same per-source train/validation indices; validation uses the same per-source CPU/CUDA seeds, resets each validation collator, and runs with zero workers. Therefore Mamba fixed-validation epochs 20 onward and ViT's fixed-validation resume from epoch 2 onward use the same examples, masks, and channel subsets. The earlier legacy validation segments remain excluded from direct comparison.

The repository preprocessing audit found no cache-generation script for these four files. The active loader reads their existing `(N,C,224,224)` tensors and casts them to float32; it does not recompute normalization. HDF5 attributes document transforms and some mean/std values but do not reveal whether those statistics used the later validation samples. This inherited preprocessing fit-scope remains unresolved; there is no evidence that it caused the observed source-specific losses, and the caches must not be normalized a second time.

### Read-only implementation audit — 2026-10-08 08:26 MDT

The latest checkpoint was read on CPU after epoch 63 completed. It confirms Mamba-2/seed 0, fixed-validation protocol, completed epoch 63 (zero-based epoch 62), global step **13,482**, and saved optimizer, two scheduler, sampler, resolved-config, and RNG states; all **19,085,440** model tensor values are finite. Non-backbone fields in the Mamba and ViT configs compare equal after removing their architecture-specific settings. Total parameters are **19,085,440** for Mamba-2 and **13,752,064** for ViT, a **38.8%** mismatch caused by the encoder choice.

Static review found no confirmed defect in the inspected optimization path: per-source losses are averaged equally across microbatches, non-finite losses/gradients stop before the optimizer and EMA update, target parameters are frozen, EMA follows the optimizer step, and context/target mask indexing uses the same helper. The data loader intentionally max-cycles smaller sources to preserve equal per-source contribution. At **13,696 samples per source per epoch**, the 798 WiFi CSI training examples are revisited about **17.2** times per epoch, compared with **4.3** for the 3,166 spectrogram examples, **2.4** for 5,636 RFP examples, and **1.0** for 13,726 NRPos Indoor examples. This imbalance in repeat exposure is a plausible contributor to source-specific behavior, not a confirmed bug; any change should be evaluated as a separate bounded validation-only trial after the matched baseline, without using downstream test results.

ViT seed 0 remains at its preserved epoch-1 checkpoint. Its console log records SIGTERM and a graceful checkpoint at the epoch-1 boundary, with no traceback. The log does not identify the sender; the earlier conversation includes the user's explicit request to stop active training for diagnosis, so the signal is consistent with that requested interruption, though the exact sending process is unrecorded. The guarded fixed-validation resume and downstream/evaluation queues remain alive but gated behind full Mamba seed-0 completion. There are **0/8** seed-0 linear-probe results. Only Mamba-2 is using the GPU. No commit or push was made.

## Protocol and selected configuration

The primary comparison uses `configs/mamba2_prose.yml` and `configs/jepa_vit_baseline.yml`: 240 epochs, four predictor layers, effective batch 256, FP32, identical four-source mixture, masks, optimizer/schedule, and matched pretraining seeds 0, 1, and 2. `--seed` is the supported CLI override; each resolved config is saved with its run. GPU jobs run sequentially. `mamba2_table1.yml` is not used.

The only configured pretraining files are `spectrogram-pretrain.h5`, `rfp.h5`, `wifi-csi.h5`, and `nrpos-indoor.h5` (24,552 examples total). All configured keys and paths passed the latest CUDA preflight. HDF5 files are read-only. The unverified `5g-csi.h5` remains excluded.

## Environment and preflight

- Repository: `/mnt/c/Users/danis/Downloads/data2/h5_staging/MultiScaleJEPA/latent-wave`, branch `master`, HEAD `330e9ef5f6eb69e728ba43b3755036cffdd086cd` at launch preparation. Working tree already contained staged implementation changes; no commit or push was made.
- Hardware/runtime verified in the preceding readiness report: WSL2 Ubuntu 24.04.5, Python 3.13.15, uv 0.12.18, PyTorch 2.10.0+cu130, CUDA 13.0, NVIDIA driver 610.60, RTX 5070 Ti (16,303 MiB, compute capability 12.0). Official Mamba-2 and causal-conv1d CUDA paths passed the bounded tests.
- Current-code four-source dry-runs passed on `cuda:0` for Mamba-2 and ViT. Predictor output was `(48, 28, 256)`. Preflight peak allocation: Mamba-2 361,396,224 bytes; ViT 98,454,016 bytes. These are preflight figures, not training peaks.
- Total parameters: Mamba-2 19,085,440 (9,908,928 trainable); ViT 13,752,064 (7,242,240 trainable). Mamba-2 has 38.8% more total parameters; the backbone is the intended primary architecture difference, but parameter counts are not matched.
- The planned output volume is outside Git at `../runs/latent-wave/full_runs/`. At preflight it had about 655 GB free.

## Run status

| Architecture | Seed | Status | Epochs / steps | Selected checkpoint | Run directory |
|---|---:|---|---|---|---|
| Mamba-2 | 0 | Running; fixed-seed validation | 70 / 240 completed; epoch 71 at global step 15,070/51,360 (90/214; latest log snapshot 10:41 MDT) | Best epoch 70, fixed validation loss 0.322346; latest checkpoint global step 14,980 | ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/ |
| ViT | 0 | Partial; graceful SIGTERM after epoch 1; guarded resume waiting | 1 / 240 completed; global step 214 | epoch 1; original checkpoint preserved; staged fixed-validation resume is waiting | ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val/ |
| Mamba-2 | 1 | Not run; queued after seed-0 comparison | 0 / 240 epochs | none | ../runs/latent-wave/full_runs/mamba2_prose_seed1/ |
| ViT | 1 | Not run; queued after matched Mamba-2 seed 1 | 0 / 240 epochs | none | ../runs/latent-wave/full_runs/vit_prose_seed1/ |
| Mamba-2 | 2 | Not run; queued after seed 1 | 0 / 240 epochs | none | ../runs/latent-wave/full_runs/mamba2_prose_seed2/ |
| ViT | 2 | Not run; queued after matched Mamba-2 seed 2 | 0 / 240 epochs | none | ../runs/latent-wave/full_runs/vit_prose_seed2/ |

No full-run architecture comparison or downstream score is reported. Do not infer a model advantage from these partial runs, their parameter counts, or smoke tests.

### Historical seed-0 measurements before the validation fix

Mamba-2 seed 0 completed epoch 1 in **1,010.324 seconds** (214 optimizer updates; 1,712 microbatches), with mean train JEPA loss **0.764172** and mean validation JEPA loss **0.503572**. Per-source train / validation losses were: spectrogram 0.749443 / 0.442144; RFP 0.771949 / 0.497015; WiFi CSI 0.763806 / 0.543162; NRPos Indoor 0.771490 / 0.531969. Contributions were 13,696 examples per source. Aggregate source throughput was **54.224 examples/s**, computed from all source contributions over full epoch wall time (including validation). Peak allocated/reserved VRAM was **2,064,175,616 / 2,686,451,712 bytes**. The first-epoch estimate is about **67.35 hours total** and **67.07 hours remaining** for 240 epochs.

Epoch 1 was the initial best validation checkpoint. The earlier PTY Ctrl-C stopped the pipeline during epoch 2; its valid latest checkpoint was epoch 1. On 2026-10-07, seed 0 resumed from that checkpoint on the idle RTX 5070 Ti. Epoch 2 completed in **1,036.456 seconds**, with mean train JEPA loss **0.394473** and stochastic validation JEPA loss **0.331557**. Epoch 3 completed in **1,013.430 seconds**, with train loss **0.333243** and validation loss **0.351129** at **54.058 source examples/s**. That original run later reached 19 completed epochs: epoch 19 train loss **0.340943**, validation loss **0.351761**, and global step **4,066**. Its best was epoch 2 under the stochastic validation protocol. It stopped during epoch 20 at logged global step 4,170; the valid latest checkpoint stayed at epoch 19. The original logs and checkpoints are preserved in their run directory.

ViT seed 0 completed epoch 1 in **278.597 seconds** (214 optimizer updates; 1,712 microbatches), with mean train JEPA loss **0.713062** and mean validation JEPA loss **0.432451**. Per-source train / validation losses were: spectrogram 0.712303 / 0.419089; RFP 0.716734 / 0.434516; WiFi CSI 0.708873 / 0.439130; NRPos Indoor 0.714340 / 0.437067. Throughput was **196.643 source examples/s**, computed over full epoch wall time including validation. Peak allocated/reserved VRAM was **2,578,249,728 / 4,009,754,624 bytes**. The first-epoch estimate is about **18.57 hours total** and **18.50 hours remaining** for 240 epochs. The SIGTERM handler saved a valid epoch-1 checkpoint and exited normally.

Current figures are generated in `../runs/latent-wave/comparison/current/`; `figure_index.md` lists every PNG and marks paper references separately from local measurements. At the 10:34 MDT plot refresh, the training-loss curve joined the original Mamba epochs 1–19 to the resumed run from epoch 20 because the optimization checkpoint and training configuration are continuous. Validation is displayed in separate protocol panels: historical Mamba/ViT validation from the pre-fix runs and the fixed-seed Mamba continuation through completed epoch 70. These values must not be compared across panels. The in-progress epoch 71 is omitted until its validation record completes. No complete matched run or downstream result exists. Runtime projections are approximate, not completed results.

## Exact foreground launch pattern

Create the run directory, then execute this in the visible VS Code terminal. `pipefail` preserves Python's exit code while `tee` writes the same unbuffered output to disk:

```bash
mkdir -p ../runs/latent-wave/full_runs/mamba2_prose_seed0
set -o pipefail; uv run python -u src/pretrain.py --config configs/mamba2_prose.yml --device cuda --seed 0 --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0 2>&1 | tee -a ../runs/latent-wave/full_runs/mamba2_prose_seed0/train_console.log
```

The matched ViT command substitutes `configs/jepa_vit_baseline.yml`, `vit_prose_seed0`, and the corresponding console log. Repeat sequentially for seeds 1 and 2. Resume a partial run with the same config, device, seed, and output directory, plus `--resume ../runs/latent-wave/full_runs/<run-name>/latest.pt`. The trainer saves `latest.pt` after each completed epoch, `best.pt` by the label-free validation JEPA loss, and a periodic epoch checkpoint every five epochs.

The exact fresh ViT seed-0 command used was:

```bash
mkdir -p ../runs/latent-wave/full_runs/vit_prose_seed0
set -o pipefail; uv run python -u src/pretrain.py --config configs/jepa_vit_baseline.yml --device cuda --seed 0 --output-dir ../runs/latent-wave/full_runs/vit_prose_seed0 2>&1 | tee -a ../runs/latent-wave/full_runs/vit_prose_seed0/train_console.log
```

Seeds 1 and 2 use the same command pattern with `--seed 1` / `--seed 2`, distinct `_seed1` / `_seed2` run directories, and the matching architecture config.

Each epoch writes a machine-readable JSON line and each approximately 5% of its optimizer steps writes a visible `[progress]` record. Those records include architecture, seed, epoch and total, step and total, rolling mean/per-source training loss, previous completed validation loss, learning rate, EMA momentum, elapsed time, ETA, GPU allocation/reservation, and per-source example contributions. An in-epoch validation value is unavailable until the first validation pass completes.

## Throughput and checkpoint measurements

| Run | Seed | Completed epochs / steps | Source examples/s | Training time | Peak VRAM | Total/trainable parameters | Best-validation epoch | Checkpoint |
|---|---:|---:|---:|---|---|---:|---:|---|
| Mamba-2 original (stochastic validation) | 0 | 19 / 4,066 | 59.984 (epoch 19) | 18,409.314 s elapsed; interrupted during epoch 20 | 2.36 GB allocated / 4.15 GB reserved | 19,085,440 / 9,908,928 | 2 (stochastic metric) | ../runs/latent-wave/full_runs/mamba2_prose_seed0/best.pt |
| Mamba-2 continuation (fixed validation) | 0 | 48 total; 29 continuation / 10,272 | 51.383 (epoch 48) | 47,604.854 s cumulative; 1,066.192 s for epoch 48 | 2.28 GiB allocated / 5.12 GiB reserved | 19,085,440 / 9,908,928 | 43 (fixed metric, 0.328130) | ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/best.pt |
| ViT original partial | 0 | 1 / 214 | 196.643 | 278.597 s / epoch; initial 18.57 h estimate for 240 | 2.58 GB allocated / 4.01 GB reserved | 13,752,064 / 7,242,240 | 1 (stochastic metric) | ../runs/latent-wave/full_runs/vit_prose_seed0/best.pt |

The original Mamba-2 segment has 19 epochs, the fixed-validation continuation has 19, and ViT has one. None is an observed full-run training time, and the two Mamba validation protocols must not be joined as one comparable curve. The latest epoch-39 progress record estimates about 54h42m remaining; this is a moving estimate.

## Downstream evaluation

Once both seed-0 pretraining runs complete, the foreground watcher `../runs/latent-wave/full_runs/run_linear_evaluations_after_pretraining.sh` uses each validation-selected `best.pt` for all four tasks (`rfs`, `pos`, `deepmimo-beam`, `deepmimo-los`) with `src/evaluate_linear.py` and split seeds 0, 1, 2. Both architectures share `../runs/latent-wave/shared_splits/controlled_seed0_fixed_validation/`. Results are written to `../runs/latent-wave/results/controlled_seed0_fixed_validation/`; the watcher then refreshes comparison artifacts. It does not start evaluation unless both models report 240 completed epochs. The evaluator writes only aggregate metrics per split and does not save predictions; confusion matrices and predicted-versus-true/error plots require extending the output format.

### Supervised fine-tuning status

The existing `src/finetune.py` cannot directly fine-tune these JEPA checkpoints: its `--model-type` choices are legacy `latentwfm`/`wavesfm` (the latter raises because its module is absent), and its initializer expects a `state_dict` plus key prefix. The JEPA-aware `src/finetune_jepa.py` loads the target encoder, reuses shared split manifests, and carves validation only from the training partition. Review found and fixed a missing resolved checkpoint-path attribute. It now has explicit `--validation-only` screening and `--test-only` evaluation modes, so candidate selection can avoid opening test data and the chosen `best.pt` can be evaluated afterward. Resume and test-only paths verify the saved split manifest, train/validation indices, pretraining checkpoint, and training settings. Syntax compilation passed; runtime/data smoke and training have not been run. A CLI-help probe stalled while importing under active-run I/O pressure and was terminated. A guarded follow-on queue is waiting until the matched seed-1/2 training and all frozen-probe outputs finish. It screens two learning rates from validation across shared split seeds 0–2, then evaluates only the selected setting on the held-out splits; the separate output summarizer creates per-class, confusion-matrix, and positioning-error artifacts.

No downstream task was launched: neither architecture has a complete 240-epoch run, so downstream scores would not fulfill the stated comparison protocol. RF classification, outdoor positioning, DeepMIMO beam prediction, and DeepMIMO LoS/NLoS are all marked **not run** in `downstream_status.csv` and `downstream_status.png`.

At **22:34:28 MDT**, Mamba-2 seed 0 was live in epoch 30 at local step **24/214** (global step **6,230/51,360**). The latest rolling train loss was **0.327411**, last validation **0.343103**, LR **9.862e-4**, EMA **0.996485**, elapsed **7h51m**, ETA about **56h52m**, and PyTorch memory **2.13/3.52 GiB allocated/reserved**. The latest complete checkpoint is epoch 29/global step 6,206; the selected best remains epoch 25.

Loaded the epoch-28 `latest.pt` read-only on CPU to verify resume integrity. It records zero-based epoch 27 (28 completed), seed 0, architecture `mamba2`, global step 5,992, validation protocol, resolved config, Git revision, optimizer state (263 entries), two scheduler states, sampler states for all four sources, and Python/NumPy/Torch/CUDA RNG states. All 19,085,440 model values across 478 tensors are finite. This verifies checkpoint contents and does not test activation collapse.

Generated artifacts are outside Git under `../runs/latent-wave/comparison/current/`: pretraining run summaries, loss traces, validation figures separated by protocol, a compute comparison, downstream status files, and paper comparison files. The downstream status figure labels all four tasks and marks them not run; there are no downstream score files.

## Paper results reference comparison

The paper's Table II reports three-seed mean ± standard deviation for LatentWave with Region and Frequency masks, WavesFM, and a supervised reference. These values are in `paper_results_reference.csv` and `../runs/latent-wave/comparison/current/paper_results_comparison.png`; the generated CSV/JSON/Markdown files are alongside the PNG. Current Mamba-2 and ViT downstream slots remain blank because evaluation has not run. The paper scores are contextual reference values under the paper's setup, not matched reruns of this repository.

## 2026-10-08 00:53 MDT — epoch 37 result and refreshed figures

Mamba-2 seed 0 completed epoch **37/240** at global step **7,918**. Mean train JEPA loss was **0.322280**, fixed-validation loss **0.335245**, duration **1,034.570 s**, throughput **52.953 source examples/s**, cumulative elapsed **36,010.495 s**, and boundary ETA **197,571 s (~54h53m)**. The validation loss is slightly higher than epoch 36 (**0.335008**); best validation remains epoch 35 (**0.332309**). These measurements do not establish a downstream gain.

The latest saved checkpoint was CPU-audited as Mamba-2 seed 0, zero-based epoch **36**, step **7,918**, fixed validation, with optimizer, scheduler, sampler, resolved config, and RNG state. All **19,085,440** model tensor values are finite. At **00:53 MDT**, the same foreground trainer PID **1149902** had started epoch 38 at step **7,930/51,360** (12/214), with rolling train loss **0.320681**, last validation **0.335245**, LR **9.705e-4**, EMA **0.996618**, ETA **197,551 s (~54h52m)**, and PyTorch allocated/reserved memory **2.13/3.52 GiB**. RTX 5070 Ti use was **5,340/16,303 MiB** at **15%** utilization.

Refreshed the external plots and machine-readable summaries through epoch 37 at `../runs/latent-wave/comparison/current/`; `figure_index.md` reports Mamba 37/240, epoch 38 underway, and ViT 1/240. The paper Table II comparison keeps local downstream result slots blank, and Table III remains labeled paper-reported reference. The ViT handoff, seed-0 evaluator, and matched-seed queue are alive and gated behind the active Mamba run. Seed-0 downstream output remains **0/8**; none of the four tasks has been evaluated.

Table III's one-seed masking ablation is transcribed from the attached PDF (arXiv v1; `latentwave.pdf`) in `paper_masking_reference.csv` and plotted in `../runs/latent-wave/comparison/current/paper_masking_ablation_reference.png`. It covers RF classification, positioning, and beam prediction; the paper does not report LoS/NLoS in that table. This is a paper-reported reference figure, not a local masking experiment. No masking ablation has been run locally.

Regenerate the figure and comparison artifacts with:

```bash
./.venv/bin/python src/plot_paper_comparison.py \
  --paper-reference docs/mamba2_experiment/paper_results_reference.csv \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --downstream-status ../runs/latent-wave/comparison/current/downstream_status.csv \
  --out-dir ../runs/latent-wave/comparison/current
```

Recreate the Table III reference and logged pretraining figures with:

```bash
./.venv/bin/python src/plot_masking_ablation_reference.py \
  --reference-csv docs/mamba2_experiment/paper_masking_reference.csv \
  --out-dir ../runs/latent-wave/comparison/current
./.venv/bin/python src/plot_training_history.py \
  --run-dirs ../runs/latent-wave/full_runs/mamba2_prose_seed0 \
             ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
             ../runs/latent-wave/full_runs/vit_prose_seed0 \
  --out-dir ../runs/latent-wave/comparison/current
./.venv/bin/python src/plot_compute_comparison.py \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --out-dir ../runs/latent-wave/comparison/current
```

## Live terminal monitor

`src/monitor_training.py` follows the two console logs in read-only mode, polls every two seconds, and redraws only when a log record or process state changes. It prints one permanent summary line per completed epoch and keeps six live status lines below them. Start it from a second VS Code WSL terminal at the repository root:

```bash
./.venv/bin/python -u src/monitor_training.py \
  --mamba-log ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/train_console.log \
  --mamba-run-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
  --vit-log ../runs/latent-wave/full_runs/vit_prose_seed0/train_console.log \
  --vit-run-dir ../runs/latent-wave/full_runs/vit_prose_seed0 \
  --refresh 2
```

The saved `[progress]` records contain a global step and total run steps/epochs; the monitor derives the local epoch step and bar from those values. They contain loss, validation loss, LR, EMA, elapsed/ETA, and PyTorch allocated/reserved VRAM. They do not contain the OS process state, so the monitor reads `/proc` to label a run as running or stopped. They also do not contain GPU utilization or free/total memory. The ViT process is stopped after its first completed epoch; its row displays the last logged values with that stopped status.

## Interpretation

No architecture comparison is valid until at least one matched full 240-epoch Mamba-2 and ViT pair completes. Report measured results alongside the parameter-count difference, full-run throughput, peak memory, and validation-selected epochs. If the session ends before all seeds finish, the remaining cells stay marked partial/not run and exact resume commands are retained. The initial deterministic-validation restart was launched with:

```bash
set -o pipefail; ./.venv/bin/python -u src/pretrain.py --config configs/mamba2_prose.yml --device cuda --seed 0 --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val --resume ../runs/latent-wave/full_runs/mamba2_prose_seed0/latest.pt --reset-validation-selection 2>&1 | tee -a ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/train_console.log
```

After the new continuation writes its epoch-20 checkpoint, resume it without resetting validation selection:

```bash
set -o pipefail; ./.venv/bin/python -u src/pretrain.py --config configs/mamba2_prose.yml --device cuda --seed 0 --output-dir ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val --resume ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/latest.pt 2>&1 | tee -a ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/train_console.log
```

The matched ViT continuation should use a separate deterministic-validation directory and reset selection once when it is started:

```bash
set -o pipefail; ./.venv/bin/python -u src/pretrain.py --config configs/jepa_vit_baseline.yml --device cuda --seed 0 --output-dir ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val --resume ../runs/latent-wave/full_runs/vit_prose_seed0/latest.pt --reset-validation-selection 2>&1 | tee -a ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val/train_console.log
```

Regenerate run tables, the partial validation-loss plot, and downstream task-status artifacts from the saved JSON logs with:

```bash
uv run python src/summarize_training_runs.py \
  --run-dirs \
    ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
    ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val \
  --out-dir ../runs/latent-wave/comparison
```

## Live update — 2026-10-07 22:38 MDT

Mamba-2 seed 0 is active at epoch 30, step 74/214 (global step 6,280/51,360). Rolling training loss is 0.328739; last completed fixed-validation loss is 0.343103 from epoch 29; LR 9.858e-4; EMA 0.996489; logged elapsed time 7h54m; ETA 56h47m; PyTorch memory 2.13/3.52 GiB allocated/reserved. The GPU is the RTX 5070 Ti (5,338/16,303 MiB used, 15% utilization in a spot sample). Metadata and the latest valid checkpoint confirm 29 completed epochs/global step 6,206; the best fixed-validation checkpoint is epoch 25 at 0.336708. Training loss fell from 0.764172 at epoch 1 to 0.330258 at epoch 29. Fixed-validation loss moved from 0.350029 at epoch 20 to 0.336708 at epoch 25, then fluctuated upward to 0.343103 at epoch 29. This indicates optimization progress with noisy validation, not a downstream performance gain. ViT seed 0 remains at its valid epoch-1 checkpoint and is queued to resume only after Mamba completes. The downstream watcher waits for both 240-epoch runs; no test scores are available. PNG figures under `../runs/latent-wave/comparison/current/` include the latest completed epoch 29 and mark downstream results unavailable.

A visible serial continuation queue is now waiting outside Git for the eight seed-0 task JSONs and an idle GPU. It will not overlap training/evaluation jobs. Once seed 0's Mamba-2 and ViT runs and downstream probes finish, it runs matched seed 1 Mamba-2 then ViT, evaluates both on all four tasks with the existing split manifests, then repeats for seed 2. The queue preserves per-run logs/checkpoints and resumes a partial run from `latest.pt`; it validates metadata identity, evaluator task/model/pretraining seed, and split seeds before accepting outputs. Queue process PID 1220532 is currently waiting and has not started GPU work. Command to resume only if this queue process is absent: `set -o pipefail; bash ../runs/latent-wave/full_runs/run_matched_seeds_1_2_after_seed0.sh 2>&1 | tee -a ../runs/latent-wave/results/controlled_pretraining_seeds_fixed_validation/matched_seed1_2_queue.log`.

## Live update — 2026-10-07 22:51 MDT

Mamba-2 seed 0 completed epoch 30 at 22:50 MDT: mean training loss 0.327695, fixed validation loss 0.344843, throughput 55.867 source examples/s, 980.61 seconds. The latest checkpoint is 30/240 at global step 6,420; best fixed-validation checkpoint is still epoch 25 at 0.336708. At 22:51, epoch 31 was at step 10/214 (global step 6,430/51,360), rolling train loss 0.329010, prior validation 0.344843, LR 9.846e-4, EMA 0.996501, logged elapsed 8h06m, ETA about 56h39m, and PyTorch allocated/reserved memory 2.13/3.52 GiB. Seed-0 ViT remains at its epoch-1 checkpoint. The evaluation watcher has written no downstream results (0/8); the seed-1/2 queue is waiting and no second GPU job is active. The plots under `../runs/latent-wave/comparison/current/` were refreshed through completed epoch 30; they do not include partial epoch 31. Downstream metrics remain unavailable.

A read-only CPU audit of the completed epoch-30 checkpoint at 22:54 MDT verified identity Mamba-2/seed 0, zero-based epoch 29/global step 6,420, and 19,085,440 finite model values across context, target, and predictor tensors. Optimizer, scheduler, sampler, and all RNG states plus resolved config/revision metadata are present. This excludes nonfinite/all-zero checkpoint weights but does not prove healthy forward activations or rule out representation collapse. At the same snapshot, the live trainer was at epoch 31 step 50/214 with rolling train loss 0.325626, last validation 0.344843, and ETA about 56h36m; no second GPU job was started.

At 22:57:36 MDT, Mamba-2 seed 0 was active at epoch 31 step 80/214 (global step 6,500/51,360), rolling training loss 0.326242, latest validation 0.344843 (epoch 30), LR 9.841e-4, EMA 0.996506, ETA 56h34m, and PyTorch allocated/reserved VRAM 2.13/3.52 GiB. The latest checkpoint is epoch 30; best validation remains epoch 25 at 0.336708. The RTX 5070 Ti used 5,343/16,303 MiB at 16% in a spot sample. ViT and downstream evaluation remain pending; no scores exist.

At 22:59:41 MDT, Mamba-2 seed 0 was active at epoch 31 step 110/214 (global step 6,530/51,360), rolling train loss 0.325078, latest validation 0.344843 (epoch 30), LR 9.838e-4, EMA 0.996509, ETA about 56h31m, and PyTorch memory 2.13/3.52 GiB allocated/reserved. The latest complete checkpoint is epoch 30; best validation remains epoch 25 at 0.336708. The ViT continuation and downstream evaluation are still pending, with no results.

## Live update — 2026-10-07 23:08 MDT

Mamba-2 seed 0 is running at 31/240 completed epochs, step 6,634/51,360. Epoch 31 logged mean train loss **0.324773**, fixed-seed validation loss **0.335272** (new best, selected checkpoint epoch 31), 55.927 source examples/s, and 2.13/3.52 GiB PyTorch peak allocated/reserved memory. Logged ETA is approximately **56h24m**. Validation improved from 0.350029 at fixed-protocol epoch 20; the rise at epoch 30 did not persist. This evidence supports continued training under the documented configuration. No downstream scores are available, and ViT remains stopped after epoch 1 awaiting its guarded sequential resume.

The current PNGs and CSV/JSON summaries were regenerated from the completed epoch-31 checkpoint/log. The figures are snapshots and will be refreshed after later completed epochs. No training code or checkpoint was changed for this refresh.

### Handoff readiness — 2026-10-07 23:11 MDT

The guarded ViT resume and downstream evaluation watcher were checked against their live paths. The ViT epoch-1 source checkpoint matches the staged fixed-validation checkpoint byte-for-byte. The handoff waits for Mamba metadata to confirm all 240 epochs before starting ViT. The evaluation watcher also waits for both fixed-validation runs to finish, then uses shared per-task split manifests for seeds 0, 1, and 2. No split/result files exist yet because evaluation has not started. The only active CUDA training process remains Mamba-2 seed 0.

### Throughput observation — 2026-10-07 23:14 MDT

At epoch 32, step 76/214, Mamba-2 seed 0 logs rolling train loss 0.326239, last completed validation loss 0.335272, and roughly 56h16m remaining. A brief read-only GPU sample measured 15–20% SM and 3–4% memory-controller utilization while the trainer used about 111% CPU; DataLoader workers were mostly idle. This suggests low GPU occupancy, but does not isolate Python launch overhead from input loading. The active training run and configuration were left unchanged.

### Epoch 32 result — 2026-10-07 23:26 MDT

Mamba-2 seed 0 completed epoch 32/240 (global step 6,848). Mean train JEPA loss was 0.328984; fixed-seed validation loss was 0.348405. The best validation checkpoint remains epoch 31 at 0.335272. The epoch-32 RFP validation component rose to 0.164031, while the checkpoint-selection rule remains the same label-free mean validation loss. PNGs and summaries were refreshed through the completed epoch-32 record. The run continues in epoch 33; ViT and downstream tasks have not started.

### Source-specific loss diagnosis — 2026-10-07 23:28 MDT

Added the external PNG `pretraining_source_losses.png` and CSV `pretraining_source_loss_traces.csv`. On the fixed validation protocol (epochs 20–32), validation loss trends upward for spectrogram (+0.000376/epoch) and RFP (+0.001294/epoch), and downward for WiFi CSI (−0.002218/epoch) and NRPos Indoor (−0.001923/epoch). Training loss trends downward for all four sources. Mean train loss fell 0.344455 to 0.328984, but the epoch-32 validation mean (0.348405) is only slightly below epoch 20 (0.350029); the selected best remains epoch 31 (0.335272). This is a descriptive source-level divergence, not proof of a specific bug. The Mamba training process remains active.

## 2026-10-07 23:31 MDT — repeat utilization sample

A second read-only `nvidia-smi dmon -s u -d 2 -c 5` sample during epoch 33 measured SM utilization **16, 17, 17, 19, 19%** and memory-controller utilization **3, 3, 3, 4, 5%**. The trainer remained at about **111% CPU** and was the only CUDA compute process. This corroborates low GPU occupancy across separate samples, but does not isolate Python launch overhead from the input path. Training was left untouched.

### Epoch 33 result — 2026-10-07 23:44 MDT

Mamba-2 seed 0 completed epoch 33/240 at step 7,062. Mean train loss was 0.325654; fixed-validation loss was 0.339354, down from 0.348405 at epoch 32. Best validation remains epoch 31 at 0.335272. The latest source-specific validation losses were 0.195561 (spectrogram), 0.144892 (RFP), 0.485658 (WiFi CSI), and 0.531306 (NRPos Indoor). Figures and summaries now include completed data through epoch 33. The process continues into epoch 34; ViT and downstream evaluation remain queued.

### Live update — 2026-10-07 23:51 MDT

Mamba-2 seed 0 is active at epoch 34, local step 108/214 (global step 7,170/51,360). Rolling train loss is 0.324319; the last completed fixed validation remains 0.339354 from epoch 33. The selected checkpoint is still epoch 31 at 0.335272; latest complete checkpoint is epoch 33. Logged ETA is about 55h40m, with 2.13/3.52 GiB PyTorch allocated/reserved and 5,336/16,303 MiB device memory in use. The ViT resume, downstream evaluation, and matched seed queue remain alive but gated; no downstream scores are available. PNGs remain current through completed epoch 33, since epoch 34 has not validated yet.

### Epoch 34 result — 2026-10-08 00:00 MDT

Mamba-2 seed 0 completed epoch 34/240 at global step 7,276. Mean train loss was 0.323856; fixed-validation loss was 0.333100, a new best versus 0.335272 at epoch 31. Epoch duration was 914.582 seconds, throughput 59.901 source examples/s, and logged ETA about 55h30m. The latest checkpoint was verified as Mamba-2 seed 0, zero-based epoch 33, with optimizer, scheduler, sampler, config, and RNG resume state; it records the fixed validation protocol and revision `330e9ef5f6eb69e728ba43b3755036cffdd086cd`. The loss curves, status panels, compute plot, and Table II reference comparison were refreshed through epoch 34. Downstream results remain unavailable and test labels have not been used for selection.

### Live update — 2026-10-08 00:01 MDT

The same Mamba-2 process remains active in epoch 35 at global step 7,300/51,360 (local step 24/214). Rolling train loss is 0.322992; most recent completed validation is 0.333100. Logged ETA is about 55h27m, LR 9.770e-4, EMA 0.996569, and PyTorch memory 2.13/3.52 GiB allocated/reserved. At 00:01:55, the RTX 5070 Ti showed 19% utilization and 5,336/16,303 MiB in use. The guarded ViT continuation, downstream evaluator, and matched-seed queue remain alive and waiting; no overlapping GPU task or downstream result exists.

The 00:03 figure refresh uses only the staged ViT seed-0 directory for current ViT plots, so its copied epoch-1 log is not drawn twice. The index labels that record as legacy validation; a distinct fixed-validation ViT curve will appear after its guarded continuation completes new epochs.

At 00:05 MDT, `src/plot_training_history.py` was updated to collapse identical copied epoch records while preserving records with distinct loss values. I reran it with the same source and staged directories used by the final queue; it collapsed the copied ViT epoch-1 record once. The refreshed validation PNG was visually checked and now shows a single legacy ViT point. This reporting-only change does not affect model training.

### Live update — 2026-10-08 00:06 MDT

Mamba-2 seed 0 is active at epoch 35, local step 74/214 (global 7,350/51,360), with rolling train loss 0.326708 and last fixed validation 0.333100 from epoch 34. The best and latest complete checkpoints are both epoch 34. Logged ETA is about 55h24m; GPU is at 21% utilization and 5,336/16,303 MiB, with PyTorch reporting 2.13/3.52 GiB allocated/reserved. The refreshed PNGs show completed epochs through 34 and downstream tasks as not run. The Vit handoff, downstream watcher, and seed-1/2 queue remain waiting; no downstream test labels or scores have been used.

### Live update — 2026-10-08 00:09 MDT

Mamba-2 seed 0 is active at epoch 35, local step 124/214 (global 7,400/51,360). Rolling train loss is 0.327382, last fixed validation is 0.333100, and the logged ETA is about 55h20m. PyTorch memory is 2.13/3.52 GiB allocated/reserved; the RTX 5070 Ti reports 16% utilization and 5,343/16,303 MiB. Epoch 34 remains the latest and best completed checkpoint. No downstream result exists yet; the guards remain alive, and the PNGs remain valid through epoch 34.

### Live update — 2026-10-08 00:04 MDT

Mamba-2 seed 0 remains active at epoch 35, local step 54/214 (global step 7,330/51,360), with rolling train loss 0.324886 and latest fixed validation 0.333100 (epoch 34, current best). ETA is about 55h25m. PyTorch memory is 2.13/3.52 GiB allocated/reserved; the RTX 5070 Ti sample is 5,336/16,303 MiB at 13%. The epoch-34 checkpoint and selected best were verified; ViT, downstream evaluation, and matched seed queues remain gated, with no downstream results yet.

### 2026-10-08 00:20 MDT — epoch 35 result and current comparison status

Mamba-2 seed 0 completed epoch 35/240 at step 7,490. Mean training JEPA loss was **0.324980** and fixed-validation JEPA loss was **0.332309**, a new best (epoch 34 was 0.333100). The validation loss has improved about **5.1%** from the first comparable fixed-validation point at epoch 20 (0.350029), although it has fluctuated substantially across epochs. This is evidence of modest pretraining improvement, not yet evidence of downstream improvement. Source-level fixed-validation trends over epochs 20–35 are positive/slightly rising for spectrogram and RFP and decreasing for WiFi CSI and NRPos Indoor; no confirmed code defect has been found.

The checkpoint audit confirms Mamba-2/seed 0, step 7,490, selected epoch 35, fixed validation protocol, dirty revision 330e9ef5f6eb69e728ba43b3755036cffdd086cd, and optimizer/scheduler/sampler/config/RNG resume state. The trainer is still active at epoch 36, step 7,530/51,360 (40/214 in the epoch), with rolling train loss 0.317858, latest validation 0.332309, LR 9.747e-4, EMA 0.996586, ETA about 55h10m, and 2.13/3.52 GiB PyTorch allocated/reserved memory. RTX 5070 Ti sample: 5,340/16,303 MiB and 19% utilization. No process was stopped or restarted.

The PNG and machine-readable summaries were refreshed at about 00:20 MDT. The figure index is ../runs/latent-wave/comparison/current/figure_index.md. Mamba/ViT downstream metrics remain unavailable: the guarded ViT continuation is waiting for Mamba's 240-epoch completion, and the evaluator has produced 0/8 seed-0 task outputs. The paper comparison therefore contains published reference values plus explicit “not evaluated” experiment slots; it is not a measured model comparison.

### 2026-10-08 00:22 MDT — checkpoint integrity and throughput review

A CPU audit of the epoch-35 best/latest checkpoint found all 19,085,440 floating model values finite. Context encoder RMS was 0.08599, target encoder RMS 0.08489, predictor RMS 0.13671, and relative context-to-target L2 distance 0.08076. This rules out a corrupt/all-zero checkpoint and identical student/target weights, but does not rule out collapsed forward representations.

Static review found repeated Python boolean/scalar conversions around per-loss and per-parameter gradient finite checks in the training loop. Such conversions can synchronize CUDA and are a plausible explanation for the observed high main-process CPU use (~111%) with low RTX 5070 Ti utilization (~17%). The cost has not been isolated with a profiler. I left the loaded training process and source unchanged; this is a throughput hypothesis, not a demonstrated cause of the validation trend. Latest live log sample: Mamba-2 seed 0 at epoch 36, global step 7,550/51,360, rolling train loss 0.318564, latest completed validation 0.332309, ETA ~55h09m. Downstream evaluation remains gated and has no result files.

### Live update — 2026-10-08 00:24 MDT

Mamba-2 seed 0 remains active at epoch 36, step 7,570/51,360 (80/214 this epoch). Latest rolling mean train loss is 0.319781; completed validation remains 0.332309 from epoch 35. LR 9.743e-4, EMA 0.996590, ETA ~55h07m, PyTorch memory 2.13/3.52 GiB allocated/reserved. The RTX 5070 Ti sample showed 17% utilization and 5,348/16,303 MiB. Latest checkpoint and plots remain at completed epoch 35; downstream evaluation remains gated with no scores.

### 2026-10-08 00:35 MDT — epoch 36 result and refreshed figures

Mamba-2 seed 0 completed epoch 36/240 at global step 7,704. Mean train JEPA loss was **0.319985**, fixed-validation loss **0.335008**, epoch duration **1,022.392 s**, and throughput **53.584 source examples/s**. Validation worsened versus epoch 35's **0.332309**; the label-free selected checkpoint remains epoch 35. The CPU checkpoint audit confirms the latest checkpoint stores epoch 36/step 7,704 and has optimizer, scheduler, sampler, resolved-config, and RNG resume state.

At 00:35:35 MDT, the same trainer was active in epoch 37 at step 7,710/51,360 (6/214 this epoch), rolling train loss 0.322890, latest validation 0.335008, LR 9.729e-4, EMA 0.996600, ETA about 55h03m, and PyTorch memory 2.13/3.52 GiB allocated/reserved. RTX 5070 Ti sample: 14% utilization, 5,338/16,303 MiB.

Refreshed the external plots and machine-readable summaries through epoch 36; the figure index now reports Mamba 36/240 and ViT 1/240. The downstream task-status figure still has no local scores, and the guarded ViT/evaluation queues remain waiting.

## 2026-10-08 01:10 MDT — epoch 38 result and refreshed figures

Mamba-2 seed 0 completed epoch **38/240** at global step **8,132**. Mean train JEPA loss was **0.324053**, fixed-validation loss **0.332679**, duration **1,035.983 s**, throughput **52.881 source examples/s**, cumulative elapsed **37,047.161 s**, and boundary ETA **196,935 s (~54h42m)**. Validation improved over epoch 37's **0.335245** but did not beat the selected epoch-35 checkpoint (**0.332309**). The CPU audit confirms the latest checkpoint stores Mamba-2/seed 0, zero-based epoch 37, step 8,132, fixed validation protocol, finite model tensors, and optimizer/scheduler/sampler/config/RNG resume state.

At **01:10 MDT**, the same trainer PID **1149902** was active in epoch 39 at step **8,150/51,360** (18/214), rolling train loss **0.321992**, previous validation **0.332679**, LR **9.681e-4**, EMA **0.996635**, ETA **196,893 s (~54h42m)**, and PyTorch allocated/reserved memory **2.13/3.52 GiB**. RTX 5070 Ti use was **5,336/16,303 MiB** at **17%** utilization.

Refreshed the external PNGs and machine-readable summaries through completed epoch 38; the figure index reports Mamba 38/240, epoch 39 underway, and ViT 1/240. Downstream scores remain unavailable, and the paper comparison keeps local result slots blank. The guarded ViT continuation, seed-0 evaluator, and matched-seed queue remain active and gated behind the current Mamba run.

## 2026-10-08 01:15 MDT — live progress; figures unchanged

At the 01:15:37 MDT log sample, Mamba-2 seed 0 PID **1149902** was active at epoch **39/240**, step **8,210/51,360** (78/214 in the epoch). Rolling train loss was **0.319370**, latest completed fixed-validation loss **0.332679** from epoch 38, LR **9.674e-4**, EMA **0.996639**, elapsed **37,426 s**, and ETA **196,703 s (~54h38m)**. PyTorch allocated/reserved memory: **2.13/3.52 GiB**; RTX 5070 Ti memory/utilization: **5,335/16,303 MiB**, **19%**. Since no new epoch has completed, the PNGs in `../runs/latent-wave/comparison/current/` remain valid through epoch 38; downstream scores are still absent.

## 2026-10-08 01:18 MDT — fixed-validation trend and current progress

Across the completed fixed-validation records, mean JEPA loss decreased from **0.350029** at epoch 20 to **0.332679** at epoch 38 (about **5.0%**); the minimum is **0.332309** at epoch 35. The average over epochs 20–24 was **0.348506**, compared with **0.333668** over epochs 34–38. The source-wise change was uneven: spectrogram 0.190764→0.199417, RFP 0.132477→0.124629, WiFi CSI 0.515054→0.482680, and NRPos Indoor 0.561820→0.523990. The loss decrease is concentrated in WiFi CSI and NRPos Indoor; it does not establish better downstream scores. No concrete training-loop defect has been confirmed by the read-only review.

At **01:18:05 MDT**, Mamba-2 seed 0 PID **1149902** was active at epoch **39/240**, step **8,240/51,360** (108/214), rolling train loss **0.319945**, latest validation **0.332679**, LR **9.671e-4**, EMA **0.996642**, ETA **196,623 s (~54h37m)**, PyTorch allocated/reserved memory **2.28/4.22 GiB**, and RTX 5070 Ti use **6,055/16,303 MiB** at **16%** utilization. The PNG snapshot remains through completed epoch 38; no downstream scores are available.

## 2026-10-08 01:28 MDT — epoch 39 result; epoch 40 active

Mamba-2 seed 0 completed epoch **39/240** at global step **8,346**. Mean train JEPA loss was **0.321717**, fixed-validation loss **0.333349**, duration **1,057.461 s**, throughput **51.807 source examples/s**, cumulative elapsed **38,105.263 s**, and boundary ETA **196,389 s (~54h33m)**. Validation was slightly worse than epoch 38's 0.332679; the selected checkpoint remains epoch 35 at **0.332309**.

The latest checkpoint audit confirms Mamba-2/seed 0, zero-based epoch **38**, step **8,346**, fixed-validation protocol, finite model tensors, and saved optimizer, scheduler, sampler, resolved config, and Python/NumPy/Torch/CUDA RNG state. Metadata confirms 39 completed epochs and agrees on the best and latest validation losses.

At **01:28:03 MDT**, the trainer was active in epoch **40**, step **8,360/51,360** (14/214), rolling train loss **0.320806**, last validation **0.333349**, LR **9.657e-4**, EMA **0.996651**, ETA **196,338 s (~54h32m)**, PyTorch allocated/reserved memory **2.28/4.22 GiB**, and RTX 5070 Ti use **6,056/16,303 MiB** at **19%** utilization. Refreshed PNGs and machine-readable summaries now include epoch 39. All downstream tasks remain not run; the guarded ViT/evaluation/seed queues are still waiting for the Mamba run to finish.

## 2026-10-08 01:33 MDT — low-utilization telemetry and live epoch 40

An eight-second `nvidia-smi dmon` sample during Mamba-2 seed 0 epoch 40 showed **15–19% SM utilization** and **3–5% memory-controller utilization**. The trainer process used about **109% CPU**; its 16 DataLoader worker processes were each below 1% CPU and sleeping. This makes a main-process synchronization/launch bottleneck more plausible than workers being saturated on HDF5 reads. Static inspection found scalar host reads for each loss group at `src/pretrain_jepa.py:590–594` and Python boolean checks over each parameter gradient at lines 601–605. Those operations can synchronize CUDA repeatedly. The sample and source inspection support this bottleneck hypothesis, but no profiler has isolated its cost. I did not alter loaded code or the active run; any training-loop change would be validated and applied only to a future matched run, with the throughput difference documented.

At **01:32:51 MDT**, PID **1149902** was active at epoch **40/240**, step **8,420/51,360** (74/214), rolling train loss **0.319311**, latest completed validation **0.333349**, LR **9.650e-4**, EMA **0.996656**, ETA **196,105 s (~54h28m)**, PyTorch allocated/reserved memory **2.28/4.22 GiB**, and device memory **6,057/16,303 MiB**. The checkpoint remains epoch 39/global step 8,346; no new epoch result or plot refresh is due yet.

## 2026-10-08 01:47 MDT — status and loss trend through epoch 40

Mamba-2 seed 0 completed epoch **40/240** at global step **8,560**. Mean training JEPA loss was **0.318634** and fixed-protocol validation loss was **0.336248**. Epoch duration was **1,046.513 s**, throughput **52.349 source examples/s**, cumulative elapsed **39,152.437 s**, and boundary ETA **195,762 s (~54h23m)**. The label-free best remains epoch **35** at **0.332309**. The current `latest.pt` stores Mamba-2/seed 0, zero-based epoch 39, step 8,560, the fixed validation protocol, and optimizer, scheduler, sampler, resolved-config, and RNG resume state.

The validation loss is uneven, but has improved overall on the fixed protocol: **0.350029 at epoch 20 → 0.336248 at epoch 40** (−3.9%); the best through epoch 40 is 0.332309 at epoch 35 (−5.1%). Mean train loss moved from **0.344455** at epoch 20 to **0.318634** at epoch 40 (−7.5%). The epoch-40 validation increase versus epoch 39 was mainly in RFP (0.133246 → 0.143403); the remaining source values were spectrogram 0.193107, WiFi CSI 0.485292, and NRPos Indoor 0.523189. This is evidence of continued pretraining progress with a noisy aggregate validation series, not evidence of downstream improvement or a confirmed training bug.

At the 01:47 MDT live sample, the trainer was in epoch **41/240**, local step **30/214** (global step 8,590), with rolling train loss **0.318037**, latest validation **0.336248**, LR **9.629e-4**, EMA **0.996669**, PyTorch memory **2.28/4.22 GiB** allocated/reserved, device use **6,062/16,303 MiB** at 17%, and ETA **195,680 s (~54h21m)**. The process and its loaded code/checkpoints remain untouched.

The updated PNG/CSV/JSON/Markdown snapshot in `../runs/latent-wave/comparison/current/` includes completed records through epoch 40. The ViT seed-0 checkpoint is still one legacy-validation epoch; its guarded fixed-validation resume waits for Mamba completion. All four downstream tasks remain not run, no test labels were used, and the downstream/evaluation and seed-1/2 queues are still waiting. The seed-0 matched comparison therefore remains incomplete and no architecture ranking is claimed.

### Source-wise trend audit — 2026-10-08 01:49 MDT

The fixed-validation logs show that the mean training loss declined for each source from epoch 20 to 40, but validation behavior differs by source:

| Source | Train loss, epoch 20 → 40 | Fixed validation, epoch 20 → 40 |
|---|---:|---:|
| Spectrogram | 0.199881 → 0.185853 | 0.190764 → 0.193107 |
| RFP | 0.143475 → 0.130331 | 0.132477 → 0.143403 |
| WiFi CSI | 0.466859 → 0.430159 | 0.515054 → 0.485292 |
| NRPos Indoor | 0.567606 → 0.528191 | 0.561820 → 0.523189 |

Aggregate validation gains are concentrated in WiFi CSI and NRPos Indoor; RFP validation is the clearest source-specific concern despite falling RFP training loss. This can reflect uneven generalization, holdout variability, or domain differences; the available records do not prove which explanation applies. It is not evidence to alter the currently running controlled baseline or select on test data. Mamba seed 0 continues in epoch 41; the epoch-40 plots remain the latest complete snapshot.

The external run-output volume had **648 GB free** at 01:50 MDT; the active Mamba continuation directory occupied about **1.1 GB**. Disk capacity is not currently limiting the checkpoint plan. The results tree still has no non-empty downstream JSON files.

At **01:51 MDT**, Mamba seed 0 was in epoch **41**, local step **80/214**, rolling train loss **0.316857**, latest validation loss **0.336248** (epoch 40), LR **9.623e-4**, EMA **0.996673**, and ETA **195,539 s (~54h19m)**. The RTX 5070 Ti was the only active CUDA device job; it showed 6,070/16,303 MiB at 17% utilization. The current plots still cover the latest completed epoch (40).

At 01:56:55 MDT, Mamba-2 seed 0 was still active in epoch 41 at local step 150/214 (global step 8,710), rolling train loss 0.317280, latest validation 0.336248, and ETA about 54h16m. Device use was 6,064/16,303 MiB at 18%; the epoch-40 checkpoint remains the latest complete one.

At **02:01:52 MDT**, Mamba-2 seed 0 completed epoch **41/240** (global step **8,774**) with mean train loss **0.317010** and fixed-validation loss **0.333044**, down from **0.336248** at epoch 40. Epoch 41 took **1,056.852 s** at **51.837 source examples/s**; cumulative elapsed time was **40,210.531 s** and the boundary ETA was **195,168 s (~54h13m)**. Its validation components were spectrogram **0.192713**, RFP **0.142249**, WiFi CSI **0.476979**, and NRPos Indoor **0.520236**. The best label-free validation checkpoint remains epoch **35** at **0.332309**. A CPU checkpoint audit confirms the latest file records Mamba-2 seed 0, zero-based epoch 40, step 8,774, the expected dirty-worktree revision, resolved config, optimizer/scheduler state, and all four sampler states.

At **02:12:01 MDT**, the process was in epoch **42/240**, global step **8,900/51,360** (126/214), with rolling train loss **0.316145**, latest validation **0.333044**, LR **9.591e-4**, EMA **0.996693**, ETA **194,734 s (~54h05m)**, and RTX 5070 Ti use **6,984/16,303 MiB** at **22%** GPU utilization. The current plots and tables were regenerated through epoch 41. The matched ViT and all downstream tasks remain incomplete; no downstream ranking can be made.

Mamba-2 seed 0 completed epoch **42/240** at global step **8,988** with mean train loss **0.316981** and fixed-validation loss **0.330321**, improving from **0.333044** at epoch 41. The new best validation checkpoint is epoch **42**. Validation components were spectrogram **0.190320**, RFP **0.137620**, WiFi CSI **0.475989**, and NRPos Indoor **0.517353**; epoch duration **1,017.798 s**, throughput **53.826 source examples/s**, and boundary ETA **194,365 s (~53h59m)**. CPU checkpoint inspection verifies the saved Mamba-2/seed-0 epoch-42 latest/best checkpoint and resume state.

The comparison snapshot and PNGs were regenerated through completed epoch 42. At **02:26:17 MDT**, Mamba was in epoch **43/240**, step **9,080/51,360** (92/214), rolling train loss **0.313938**, latest validation **0.330321**, LR **9.567e-4**, EMA **0.996707**, ETA **194,015 s (~53h54m)**, and GPU memory use **6,986/16,303 MiB**. The ViT run and downstream evaluations remain queued; the local downstream comparison is not available yet.

On the common fixed-validation protocol, epoch 20 to epoch 42 mean training JEPA loss declined **0.344455→0.316981** (−8.0%) and validation loss declined **0.350029→0.330321** (−5.6%). Per-source training losses declined for all four pretraining datasets. Validation changes were spectrogram **0.190764→0.190320** (−0.2%), RFP **0.132477→0.137620** (+3.9%), WiFi CSI **0.515054→0.475989** (−7.6%), and NRPos Indoor **0.561820→0.517353** (−7.9%). This indicates aggregate JEPA validation improvement, with a specific RFP gap; it is not evidence of downstream improvement or superiority to ViT.

### Live status — 2026-10-08 02:27 MDT

Mamba-2 seed 0 remains active at epoch **43/240**, step **9,110/51,360** (122/214), with rolling train loss **0.315345**, latest completed validation **0.330321**, LR **9.563e-4**, EMA **0.996710**, elapsed **41,814 s**, and ETA **193,922 s (~53h52m)**. PyTorch allocated/reserved VRAM is **2.28/5.12 GiB**; RTX 5070 Ti use is **6,990/16,303 MiB** at **16%**. No new epoch/checkpoint or downstream result has completed since the previous snapshot. Existing PNGs are current through Mamba epoch 42 and label the runs as partial.

### Queue and stop audit — 2026-10-08 02:29 MDT

The only CUDA compute process is the active Mamba-2 seed-0 trainer, PID **1149902**; it is at epoch **43**, step **9,110/51,360**. The live monitors remain active. ViT handoff PID **1182826** is gated on Mamba completion; the linear-evaluation queue PID **1186826** is gated on both full seed-0 runs; the matched seed-1/2 queue PID **1220532** is gated on all eight seed-0 evaluation files; and fine-tuning queue PID **1301299** is gated on matched runs. There are currently no downstream result JSONs. ViT’s saved log confirms a SIGTERM followed by a graceful epoch-1 checkpoint; it does not record the sender.

### Mamba epoch 43 — 2026-10-08 02:38 MDT

Mamba-2 seed 0 completed **43/240** epochs at global step **9,202**. Mean train loss was **0.316803** and fixed-protocol validation loss **0.328130**, improving from epoch 42's **0.330321** and becoming the selected best. The best checkpoint audit confirms its architecture, seed, epoch/step, fixed-validation protocol, resolved config, code revision, optimizer/scheduler state, and sampler states. Epoch 43 took **1,061.893 s** at **51.591 source examples/s**. At the latest live sample, Mamba was at epoch **44**, step **9,210/51,360**, train loss **0.314666**, last validation **0.328130**, PyTorch VRAM **2.28/5.12 GiB**, RTX 5070 Ti use **6,989/16,303 MiB**, and ETA about **53h49m**.

The comparison PNGs and figure index were refreshed through completed epoch 43. Validation is shown in separate protocol panels; legacy and fixed-protocol values are not conflated. The Table II plot remains paper-reported references plus explicit not-evaluated local slots. All four local downstream tasks remain not run. See `../runs/latent-wave/comparison/current/figure_index.md` for the complete PNG inventory and data labels.

On the common fixed-validation protocol, epoch 20→43 mean training JEPA loss decreased **8.0%** (**0.344455→0.316803**) and validation loss decreased **6.3%** (**0.350029→0.328130**). Spectrogram validation is nearly flat (+0.15%), while RFP, WiFi CSI, and NRPos Indoor validation decreased by **1.69%**, **7.77%**, and **8.12%**, respectively. This shows a continuing aggregate JEPA validation gain with uneven source response. It does not establish downstream improvement or a Mamba-vs-ViT win.

At **02:39 MDT**, Mamba-2 seed 0 was active in epoch **44**, step **9,230/51,360** (28/214), rolling train loss **0.313798**, latest validation **0.328130**, LR **9.547e-4**, EMA **0.996719**, ETA **~53h47m**, and GPU use **6,989/16,303 MiB**. No new epoch completed, so figures remain through epoch 43; ViT and all downstream evaluations remain incomplete.

At **02:47 MDT**, the active Mamba-2 seed-0 run was at epoch **44**, step **9,330/51,360**, rolling train loss **0.316126**, latest validation **0.328130**, and ETA **~53h43m**. GPU use was **6,989/16,303 MiB**; plots remain current through completed epoch 43. The ViT resume and downstream queues remain gated behind completion of Mamba seed 0.

### Epoch 44 checkpoint and curve update — 2026-10-08 02:56 MDT

Mamba seed 0 completed **44/240** epochs at step **9,416**. Training JEPA loss was **0.316559** and fixed validation **0.330416**; this is higher than epoch 43, so the selected best remains epoch 43 at **0.328130**. Latest-checkpoint metadata and resume-state audit passed for architecture, seed, epoch/step, config/revision, optimizer, schedulers, and samplers. The PNGs, CSV/JSON summaries, and figure index are refreshed through epoch 44. At the latest live sample, epoch 45 was at step **9,430**, rolling train loss **0.315971**, latest validation **0.330416**, and ETA about **53h39m**. The downstream comparison remains not evaluated; ViT is still awaiting its guarded continuation.

Across fixed-protocol epochs 20→44, aggregate train and validation JEPA losses fell **8.1%** and **5.6%**. Source-wise, WiFi CSI, NRPos Indoor, and RFP validation decreased, while spectrogram validation increased **3.8%**. This describes JEPA holdout behavior only.

At **02:59 MDT**, Mamba seed 0 was active at epoch **45**, step **9,470/51,360**, rolling training loss **0.314836**, latest validation **0.330416**, and ETA about **53h37m**. The GPU showed **6,989/16,303 MiB** in use; there is no new epoch or downstream score, so the PNG snapshot remains through epoch 44.

At **03:06 MDT**, the active Mamba run was at epoch **45**, global step **9,560/51,360** (144/214), rolling train loss **0.317474**, latest validation **0.330416**, and ETA **~53h32m**. RTX 5070 Ti memory use was **6,989/16,303 MiB**. No epoch boundary occurred; PNGs remain through completed epoch 44.

### Epoch 45 checkpoint and figure update — 2026-10-08 03:13 MDT

Mamba seed 0 completed **45/240** epochs at step **9,630**. Mean train loss was **0.318331** and fixed validation **0.331827**, worse than epoch 44; the best remains epoch 43 at **0.328130**. The latest checkpoint audit confirms full resume state. Refreshed PNGs and tables include epoch 45, with legacy and fixed validation shown separately. At the latest live sample, epoch 46 was at step **9,640**, rolling train loss **0.325699**, latest validation **0.331827**, and ETA about **53h30m**. No downstream result JSONs are available.

From epoch 20 to 45 on the common fixed protocol, aggregate training and validation loss decreased **7.6%** and **5.2%**. Validation improved on WiFi CSI and NRPos Indoor, while spectrogram and RFP are above their epoch-20 values. No cause or downstream effect is inferred from this loss pattern.

### Live status and ViT stop audit — 2026-10-08 03:14 MDT

Mamba-2 seed 0 PID **1149902** remains active at epoch **46/240**, step **9,650/51,360** (20/214 in the epoch). Rolling train loss is **0.319716**; the latest completed fixed-validation loss remains **0.331827** from epoch 45. LR is **9.490e-4**, EMA **0.996752**, elapsed **44,553 s**, and ETA **192,571 s (~53h29m)**. PyTorch allocated/reserved memory is **2.28/5.12 GiB**; the RTX 5070 Ti reports **6,991/16,303 MiB** at **15%** utilization. Metadata confirms **45** completed epochs at step **9,630**; the current epoch has not completed, so the figures remain through epoch 45.

The ViT seed-0 console log records **SIGTERM**, then a graceful checkpoint at epoch **1/240**, step **214**; it does not identify the signal sender. No ViT trainer is currently active. Its guarded handoff waits for Mamba to confirm 240 epochs, then waits for GPU idleness before resuming ViT from the saved checkpoint. Linear probes, matched seeds, and fine-tuning remain gated, with **0/8** seed-0 downstream result files. No process or checkpoint was changed during this inspection.

### Throughput diagnosis — 2026-10-08 03:17 MDT

At **03:17:35 MDT**, Mamba seed 0 was at epoch **46**, step **9,680/51,360**, rolling train loss **0.317633**, latest fixed-validation loss **0.331827**, LR **9.485e-4**, EMA **0.996754**, and ETA about **53h28m**. The trainer used about one CPU core while the GPU utilization samples varied **14–25%**. A 5-second `/proc` sample showed no increase in physical bytes read, but that brief observation is not a profiler and does not rule out cached or intermittent data-loader work.

Static code inspection identified repeated host synchronization as a possible throughput cost: `compute_loss_groups` calls CUDA `torch.unique(...).tolist()` and loops over channel groups; the train loop performs tensor finiteness checks and `float(loss.detach())` inside that group loop. Given microbatch 8, up to 8 channel groups per source microbatch and 256 group calls per optimizer update are possible (not measured). Mamba epoch 45 logged **50.67 source examples/s** versus **196.64** for the single completed ViT epoch. This is evidence of a throughput gap and a candidate host-dispatch cause, not proof of a bottleneck or a loss-quality issue. Fixing it safely would require preserving the per-example channel-selection objective and checking numerical parity; the active run's loaded code and checkpoints were left unchanged.

Aggregate fixed-validation loss decreased **5.2%** from Mamba epoch 20 to 45, with the lowest completed value at epoch 43 (**0.328130**) and higher values at epochs 44–45 (**0.330416**, **0.331827**). Thus the measured JEPA loss is improving overall but has recent regressions; this is not downstream evidence. The PNG snapshot remains through completed epoch 45, and downstream evaluation has not begun.

At **03:20 MDT**, a runtime source snapshot was saved outside Git beside the active Mamba run: `../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/runtime_source_snapshot.tar.gz`; its manifest is `runtime_source_snapshot.json`. It contains current source, configs, dependency files, and Python version, but no data, checkpoints, or virtual environment. SHA-256 is `c54232ea620a329efb54c552b969f43983ecac6b11eff92311a967fe980e7663`. This records the dirty tree's runtime files for reproduction while keeping the active process unchanged.

### Matched-config and parameter-count audit — 2026-10-08 03:27 MDT

The Mamba and ViT configs match on dataset mixture, masks, optimization/schedule, effective batch, epochs, and seed. Common encoder settings are 224×224 input, 16-pixel patches, width 256, depth 8, MLP ratio 4, and zero drop path; predictor config is identical. Mamba uses its configured bidirectional SSM blocks, so parameter capacity is larger even though the training protocol matches.

Checkpoint tensor shapes give Mamba-2 **9,101,248** student-encoder parameters, **9,101,248** target-encoder parameters, **882,944** predictor parameters, **19,085,440** total, and **9,908,928** trainable. ViT has **6,434,560** student-encoder parameters, **6,434,560** target-encoder parameters, the same **882,944** predictor parameters, **13,752,064** total, and **7,242,240** trainable. Mamba has **41.4%** more student-encoder parameters and **38.8%** more total JEPA parameters. Trainable counts exclude the frozen target and fixed positional embeddings.

The prior compute summary left parameter and peak-memory fields empty because it read only optional metadata fields. The summarizer now reads parameter tensor shapes from the checkpoint, peak allocated/reserved VRAM from epoch logs, and converts the checkpoint's zero-based best epoch to a one-based displayed epoch. Regenerated the CSV/JSON/Markdown pretraining summaries, validation/status figures, and compute PNG. The compute figure now displays total and trainable counts plus recorded peak memory; it remains a partial-run snapshot through Mamba epoch 45 and ViT epoch 1.

### Mamba epoch 46 — 2026-10-08 03:30 MDT

Mamba seed 0 completed **46/240** epochs at step **9,844**: mean train JEPA loss **0.319550**, fixed validation **0.333810**, epoch time **1,038.270 s**, throughput **52.765 source examples/s**, cumulative elapsed **45,491.731 s**, ETA **191,856.429 s (~53h18m)**. The best label-free validation checkpoint remains epoch **43** at **0.328130**. Epoch-46 validation components were spectrogram **0.197856**, RFP **0.152001**, WiFi CSI **0.467755**, and NRPos Indoor **0.517629**.

Regenerated measured training, protocol-separated validation, source-loss, compute, downstream-status, and paper-reference PNGs and machine-readable summaries through Mamba epoch **46**. The local downstream slots remain explicitly unevaluated. Epoch-46 fixed-protocol aggregate validation is **4.6%** below epoch 20's value, but it regressed from both epoch 45 and the epoch-43 selected best; improvements are source-specific and not evidence of downstream gains.

At **03:30:45 MDT**, Mamba PID **1149902** was in epoch **47**, step **9,850/51,360**, train loss **0.319594**, last validation **0.333810**, ETA **~53h17m**, PyTorch memory **2.28/5.12 GiB allocated/reserved**, and RTX 5070 Ti use **6,986/16,303 MiB** at **17%**. ViT remains checkpointed after epoch 1; its guarded resume and all downstream queues are still waiting.

### Downstream queue readiness — 2026-10-08 03:32 MDT

Verified the evaluator task IDs and queue arguments against `src/evaluate_linear.py`: `rfs`, `pos`, `deepmimo-beam`, and `deepmimo-los`, with seeds 0/1/2. The configured HDF5 files exist (`rfs.h5` 785 MB, `nrpos-outdoor.h5` 8.9 GB, `deepmimo.h5` 121 MB), and the loaders open them read-only. The shared split folder is absent because it is generated on the first task evaluation; each per-task manifest persists the exact seed-0/1/2 train/test indices and is checked against sample count and label hash on reuse by ViT. Both architectures use the same split directory. The evaluator queue is still gated on both complete seed-0 runs; there are **0/8** result files. A `--help` attempt stalled during CPU dependency import and was stopped before it launched any evaluation; parser and queued arguments were inspected directly.

At **03:32:10 MDT**, Mamba-2 seed 0 PID **1149902** was active at epoch **47/240**, step **9,870/51,360**, rolling train loss **0.318884**, latest validation **0.333810**, LR **9.458e-4**, EMA **0.996769**, and ETA **~53h16m**. RTX 5070 Ti use was **6,995/16,303 MiB** at **19%**; PyTorch allocated/reserved memory was **2.28/5.12 GiB**. Figures remain through completed epoch 46.

At **03:34:43 MDT**, Mamba was active at epoch **47**, step **9,900/51,360**, rolling train loss **0.317976**, latest validation **0.333810**, ETA **~53h14m**, and RTX 5070 Ti use **6,995/16,303 MiB** at **19%**. No new epoch or validation result has completed; the PNG snapshot remains through epoch **46**.

At **03:36:50 MDT**, Mamba seed 0 was active at epoch **47**, step **9,920/51,360**, rolling train loss **0.318647**, latest validation **0.333810**, and ETA **~53h13m**. RTX 5070 Ti use was **6,989/16,303 MiB** at **17%**; PyTorch allocated/reserved memory was **2.28/5.12 GiB**. Figures remain current through completed epoch **46**.

### Mamba epoch 47 and refreshed PNGs — 2026-10-08 03:47 MDT

Mamba seed 0 completed epoch **47/240** at step **10,058**. Mean training loss was **0.319246**, fixed validation loss **0.332075**, epoch time **1,045.581 s**, throughput **52.396 source examples/s**, cumulative elapsed **46,538 s**, and boundary ETA **191,103 s (~53h04m)**. The validation score improved from epoch 46 (**0.333810**) but remains above the best epoch **43** (**0.328130**). At **03:47 MDT**, the run was active at epoch **48**, step **10,070/51,360**, rolling train loss **0.322201**, most recent validation **0.332075**, and RTX 5070 Ti use **6,989/16,303 MiB** (18%); PyTorch memory was **2.28/5.12 GiB**.

Refreshed PNGs and current run summaries from completed logs through epoch **47**. The compute figure contains one entry per current seed-0 model run; Mamba's initial 19-epoch checkpoint segment is included only in the training-history curves, not counted as an independent model. `figure_index.md` records the update. The validation figures label legacy versus fixed-seed protocols per epoch. The local downstream slots remain unevaluated, and there are **0/8** seed-0 task/model JSONs. Both the matched comparison and downstream evaluation remain incomplete.

The reporting code now averages split-seed scores within each pretraining checkpoint, then weights checkpoint means equally. It reports pretraining-seed SD and within-checkpoint split-seed SD separately. The current one-checkpoint error bar is labeled as split-seed SD; it is not an estimate of pretraining-seed variability. The helper has not yet processed evaluator outputs because none exist.

At **03:50 MDT**, Mamba seed 0 remained active in epoch **48/240**, step **10,090/51,360**, with rolling train loss **0.321227**, latest fixed validation **0.332075**, ETA **190,955 s (~53h02m)**, and RTX 5070 Ti use **6,989/16,303 MiB** at **17%**. The refreshed PNG snapshot remains current through completed epoch **47**; no downstream result JSON has appeared.

### CPU checkpoint integrity audit — 2026-10-08 03:51 MDT

Loaded the active run's `latest.pt` on CPU between epoch boundaries. It confirms Mamba-2/seed 0, completed epoch **47** (zero-based checkpoint epoch 46), global step **10,058**, best fixed-validation loss **0.328130** at epoch 43, and last validation **0.332075**. All **19,085,440** model-state values and **789** floating optimizer-state tensors were finite. Context and target encoder state names match **213/213**, with **9,101,248** elements each; their relative parameter L2 difference is **0.0898**. This checks checkpoint and EMA state consistency, but does not measure activation collapse or feature quality. No such diagnostics are currently logged, and the active trainer prevents a concurrent GPU feature pass. At **03:53 MDT**, Mamba was at epoch **48**, step **10,130/51,360**, rolling train loss **0.320403**, validation **0.332075**, and ETA **190,818 s (~53h00m)**. The downstream result count remains **0/8**.

### Live progress update — 2026-10-08 03:54 MDT

Mamba-2 seed 0 PID **1149902** is verified active in epoch **48**, global step **10,150/51,360** (92/214 this epoch), rolling mean training loss **0.320053**, and latest completed fixed-validation loss **0.332075** (epoch 47). LR is **9.417e-4**, EMA is **0.996790**, elapsed time is **46,981 s**, and ETA is **190,749 s (~52h59m)**. PyTorch allocated/reserved VRAM is **2.28/5.12 GiB**; the RTX 5070 Ti reports **6,995/16,303 MiB** at **15%** utilization. The latest completed checkpoint remains epoch 47 at global step 10,058; best fixed-validation checkpoint remains epoch 43 at **0.328130**. The ViT resume and downstream/evaluation queues remain gated, and result files remain **0/8**. Existing PNGs show completed data through epoch 47 and were generated at 03:48 MDT; they do not show in-epoch progress. No training process or checkpoint was modified.

### Live progress update — 2026-10-08 03:56 MDT

Mamba-2 seed 0 PID **1149902** remained active at epoch **48**, step **10,170/51,360** (112/214 this epoch), with rolling training loss **0.319297** and last completed fixed validation **0.332075**. LR is **9.413e-4**, EMA **0.996792**, elapsed **47,077 s**, and ETA **190,667 s (~52h58m)**. PyTorch memory is **2.28/5.12 GiB** allocated/reserved; device memory is **6,992/16,303 MiB** at **16%**. The latest checkpoint is still epoch 47/global step 10,058 (mtime 03:47); no epoch boundary has occurred. Queue logs still show Mamba at 47 completed epochs, ViT stopped after epoch 1, and **0/8** downstream outputs. Existing PNGs remain current through epoch 47.

### Live progress update — 2026-10-08 03:58 MDT

After a one-minute poll of the same live trainer, Mamba-2 seed 0 PID **1149902** advanced to epoch **48**, step **10,200/51,360** (142/214 this epoch). Rolling training loss is **0.319231**; the latest completed fixed-validation loss remains **0.332075** from epoch 47. LR is **9.409e-4**, EMA **0.996794**, elapsed **47,226 s**, and ETA **190,569 s (~52h56m)**. PyTorch allocated/reserved VRAM is **2.28/5.12 GiB**; device use is **6,989/16,303 MiB** at **16%**. No epoch boundary occurred, the latest checkpoint remains epoch 47/global step 10,058, and no downstream result files appeared. The PNGs therefore remain current through epoch 47.

### Epoch 48 and plot refresh — 2026-10-08 04:05–04:06 MDT

Mamba-2 seed 0 completed **48/240** epochs at step **10,272**: mean training loss **0.318575**, fixed validation loss **0.332120**, epoch time **1,066.192 s**, throughput **51.383 source examples/s**, cumulative elapsed **47,604.854 s**, and boundary ETA **190,419.417 s (~52h54m)**. The best remains epoch 43 (**0.328130**). The epoch-48 result is slightly worse than epoch 47 (**0.332075**). The checkpoint was verified read-only on CPU: Mamba-2, seed 0, zero-based epoch 47, global step 10,272, fixed-validation protocol, optimizer/scheduler/sampler and RNG states present.

At **04:06 MDT**, Mamba was active in epoch **49**, step **10,280/51,360**, rolling train loss **0.315925**, last validation **0.332120**, LR **9.397e-4**, EMA **0.996801**, elapsed **47,646 s**, ETA **190,398 s (~52h53m)**, PyTorch VRAM **2.28/5.12 GiB**, and device VRAM **6,987/16,303 MiB** at **16%**. The latest validation curve now has 48 fixed-protocol points (epochs 20–48), with the legacy protocol kept separate.

Refreshed the PNGs and machine-readable summaries in `../runs/latent-wave/comparison/current/` through completed epoch 48 and updated `figure_index.md`. The plots include training loss, protocol-separated validation, per-source losses, compute, downstream status, and paper Table II comparison. The paper Table II values remain references, not local scores. All four downstream tasks remain not run; result count is **0/8**. The ViT resume and evaluator queue remain gated pending full 240-epoch runs. No complete model comparison or downstream result is available.

At **04:08 MDT**, Mamba seed 0 was active in epoch **49**, step **10,310/51,360** (38/214), rolling train loss **0.317448**, latest fixed validation **0.332120**, LR **9.392e-4**, EMA **0.996803**, elapsed **47,795 s**, ETA **190,299 s (~52h52m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti device use **6,987/16,303 MiB** at **17%**. The latest checkpoint is epoch 48/global step 10,272. No new epoch or downstream result has completed since the PNG refresh; figures remain current through epoch 48.

### Host/GPU utilization sample — 2026-10-08 04:10 MDT

A read-only five-sample observation of the active Mamba PID **1149902** showed **100.5–110% CPU** for the trainer (roughly one core), while the host was **94–95% idle** with no I/O wait. NVIDIA-SMI SM utilization was **18–20%** and memory-controller utilization **3–6%**. This supports a likely Python dispatch/host-synchronization limitation, especially given the GPU `.tolist()`, tensor-to-bool, and `float(loss.detach())` operations in the per-channel-group training path. It is evidence of underused hardware, not proof of which operation dominates. The active source and process remain unchanged pending the matched baseline. At the same time, the run was at epoch **49**, step **10,330/51,360**, train loss **0.317205**, validation **0.332120**, ETA **190,214 s (~52h50m)**, and GPU memory **6,988/16,303 MiB**. No downstream outputs are available.

At **04:13 MDT**, Mamba seed 0 remained active at epoch **49**, step **10,370/51,360** (98/214), with rolling train loss **0.316870**, last validation **0.332120**, LR **9.383e-4**, EMA **0.996808**, elapsed **48,075 s**, ETA **190,028 s (~52h47m)**, PyTorch memory **2.28/5.12 GiB**, and device memory **6,984/16,303 MiB** at **18%**. The latest checkpoint remains epoch 48/global step 10,272. No new checkpoint, validation result, or downstream output appeared; existing plots remain current through epoch 48.

At **04:15 MDT**, the active Mamba run was at epoch **49**, step **10,400/51,360**, rolling train loss **0.315903**, last completed validation **0.332120**, LR **9.378e-4**, EMA **0.996810**, elapsed **48,214 s**, ETA **189,889 s (~52h45m)**, PyTorch memory **2.28/5.12 GiB**, and GPU use **6,990/16,303 MiB** at **16%**. The valid latest checkpoint remains epoch 48/global step 10,272. No new validation or downstream result is available; figures remain through epoch 48.


### Mamba seed 0: epoch 49 complete — 2026-10-08 04:23 MDT

Mamba-2 seed 0 completed **49/240** epochs at step **10,486/51,360**. Mean train JEPA loss was **0.316659** and fixed-validation loss was **0.331619**, slightly below epoch 48's **0.332120**. Epoch 49 took **1,037.738 s** at **52.792 source examples/s**. The selected checkpoint remains epoch 43 with validation loss **0.328130**. From epoch 20 to epoch 49, mean training loss decreased from **0.344455** to **0.316659** (about **8.1%**) and fixed validation loss from **0.350029** to **0.331619** (about **5.3%**). This is evidence of pretraining-loss improvement with a recent plateau; it does not establish improved downstream performance.

At **04:23 MDT**, the same trainer PID **1149902** was active at epoch **50**, step **10,500/51,360**, rolling train loss **0.318326**, most recent completed validation loss **0.331619**, LR **9.363e-4**, EMA **0.996818**, elapsed **48,712 s**, and ETA **189,561 s (~52h39m)**. PyTorch memory was **2.28/5.12 GiB** allocated/reserved; the RTX 5070 Ti reported **6,981/16,303 MiB** at **17%** utilization. The four configured pretraining sources each contributed 13,696 examples in completed epoch 49; `5g-csi.h5` remains excluded. The ViT run is still stopped at epoch 1, and the seed-0 evaluation queue remains gated with **0/8** downstream result files.

Refreshed the training, validation, source-loss, compute, downstream-status, and paper-comparison figures from run logs and metadata at **04:23 MDT**. The current PNGs and machine-readable summaries are under `../runs/latent-wave/comparison/current/`; `figure_index.md` records the plot meanings and source labels. The locally measured curves include completed training through epoch 49, while paper bars remain paper-reported references. The exact commands are recorded in `06_implementation_log.md`.

### Checkpoint audit and live status — 2026-10-08 04:29 MDT

Read the latest checkpoint on CPU while the active trainer continued. It records Mamba-2, seed 0, zero-based checkpoint epoch **48** (49 completed), global step **10,486**, fixed validation protocol `fixed_seed_masks_channels_each_example_once_v1`, best checkpoint epoch **43** with validation loss **0.328130**, and last validation loss **0.331619**. All **19,085,440** model tensor values are finite. Optimizer, schedulers, sampler state, and Python/NumPy/PyTorch/CUDA RNG states are present. The checkpoint's code revision is **330e9ef5f6eb69e728ba43b3755036cffdd086cd** on dirty `master`; it is the same base revision recorded previously.

At **04:29 MDT**, Mamba seed 0 PID **1149902** was active at epoch **50/240**, step **10,570/51,360** (84/214), rolling train loss **0.318915**, and latest completed validation **0.331619**. LR was **9.352e-4**, EMA **0.996823**, elapsed **49,055 s**, ETA **189,305 s (~52h35m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti memory **6,980/16,303 MiB** at **21%** utilization. The latest checkpoint remains the completed epoch-49 state. The ViT handoff, seed-0 evaluator, matched-seed queue, and fine-tuning queue processes remain present and gated; no downstream result files exist. No training process or checkpoint was changed.

### Source-wise validation review — 2026-10-08 04:31 MDT

Compared fixed-protocol per-source validation loss at epochs 20 and 49 from `pretraining_source_loss_traces.csv`. NRPos Indoor decreased **8.55%** (**0.561820→0.513776**), WiFi CSI decreased **8.19%** (**0.515054→0.472865**), and spectrogram was nearly flat (**−0.82%**, **0.190764→0.189208**). RFP increased **13.70%** (**0.132477→0.150626**); it was also better at epoch 43 (**0.130233**) than at epoch 49. The aggregate JEPA validation improvement is therefore mixed across sources and is mainly carried by NRPos Indoor and WiFi CSI. Logged source contributions are equal at **13,696 examples per source per completed epoch**, so these measurements do not indicate a sample-count imbalance. RFP's regression could reflect source-specific fit or validation variation; this evidence alone does not establish its cause. No data mixture, loss weighting, checkpoint-selection rule, or active process was changed. The source-loss PNG already reflects these epoch-49 values.

### Mamba live progress — 2026-10-08 04:34 MDT

Mamba-2 seed 0 PID **1149902** is still active at epoch **50/240**, step **10,630/51,360** (144/214 this epoch). Rolling mean training loss is **0.317851**; latest completed fixed-validation loss remains **0.331619** from epoch 49. LR is **9.342e-4**, EMA **0.996828**, elapsed time **49,382 s**, ETA **189,213 s (~52h34m)**, PyTorch allocated/reserved memory **2.28/5.12 GiB**, and RTX 5070 Ti use **6,984/16,303 MiB** at **15%** utilization. The latest checkpoint remains epoch 49/global step 10,486, best validation remains epoch 43, and the current figures remain through epoch 49. ViT resume and downstream queues remain gated; there are no seed-0 downstream result JSONs. No job or checkpoint was changed.

### Mamba epoch 50 and refreshed plots — 2026-10-08 04:41 MDT

Mamba-2 seed 0 completed **50/240** epochs at step **10,700**. Mean training JEPA loss was **0.315172**; fixed-protocol validation loss was **0.324860**, a new best (previous best: epoch 43, **0.328130**). Epoch 50 took **1,087.552 s** at **50.374 source examples/s**. A CPU audit of `latest.pt` confirms Mamba-2/seed 0, zero-based epoch **49** (50 completed), step **10,700**, best validation **0.324860**, the fixed validation protocol, all **19,085,440** model values finite, and optimizer/scheduler/sampler/RNG resume state present.

At **04:41 MDT**, PID **1149902** continued in epoch **51**, step **10,710/51,360**, with rolling train loss **0.306138**, last validation **0.324860**, LR **9.329e-4**, EMA **0.996834**, elapsed **49,779 s**, ETA **188,936 s (~52h29m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti use about **6,984/16,303 MiB**. No other GPU training process is active. The ViT handoff and evaluation queues remain gated; seed-0 downstream results remain **0/8**.

At epoch 50, all four source validation losses decreased relative to epoch 49. Compared with epoch 20, NRPos Indoor is down **8.94%** (**0.561820→0.511572**), WiFi CSI down **8.69%** (**0.515054→0.470310**), spectrogram down **2.99%** (**0.190764→0.185053**), and RFP is nearly unchanged (**+0.02%**, **0.132477→0.132505**). The temporary RFP worsening at epoch 49 did not persist at epoch 50. These are pretraining validation losses only; downstream scores are not available.

Refreshed the PNGs and machine-readable summaries in `../runs/latent-wave/comparison/current/` through completed epoch 50 and updated `figure_index.md`. The paper comparison remains explicitly labeled as reference values and local downstream tasks remain not evaluated. The same `summarize_training_runs.py`, `plot_training_history.py`, `plot_compute_comparison.py`, and `plot_paper_comparison.py` commands recorded in the epoch-49 entry were rerun successfully with the same run directories and output directory.

### Mamba in-epoch status — 2026-10-08 04:44 MDT

PID **1149902** remains active at epoch **51/240**, step **10,740/51,360** (40/214 in the epoch). Rolling train loss is **0.311275** and latest completed validation remains **0.324860** from epoch 50. LR is **9.324e-4**, EMA **0.996836**, elapsed time **49,924 s**, ETA **188,818 s (~52h27m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti memory use **6,983/16,303 MiB** at **15%** utilization. The latest checkpoint remains completed epoch 50, with the best validation checkpoint at epoch 50. No additional GPU job or downstream evaluation has started; figures remain current through completed epoch 50.

### Mamba in-epoch status — 2026-10-08 04:47 MDT

Mamba-2 seed 0 PID **1149902** remains active at epoch **51/240**, step **10,780/51,360** (80/214 in the epoch). Rolling mean train loss is **0.312371**; latest completed fixed validation is **0.324860** from epoch 50. LR is **9.318e-4**, EMA **0.996840**, elapsed time **50,122 s**, ETA **188,678 s (~52h25m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti memory use **6,986/16,303 MiB** at **15%** utilization. The latest checkpoint is still epoch 50. The ViT resume and evaluation queues remain waiting, with **0/8** downstream outputs.

### Mamba epoch 51 and refreshed plots — 2026-10-08 04:59 MDT

Mamba-2 seed 0 completed **51/240** epochs at step **10,914**. Mean train loss was **0.311242**, fixed-validation loss **0.326645**, epoch time **1,066.798 s**, throughput **51.354 source examples/s**, and cumulative elapsed time **50,800.190 s**. The best remains epoch **50** at **0.324860**. A CPU audit of the updated latest checkpoint confirms Mamba-2/seed 0, zero-based epoch **50** (51 completed), global step **10,914**, fixed validation protocol, all **19,085,440** model values finite, and optimizer/scheduler/sampler/RNG state present. Its last validation is **0.326645** and best validation is **0.324860**.

At **04:59 MDT**, PID **1149902** continued in epoch **52**, step **10,930/51,360**, rolling train loss **0.313643**, latest completed validation **0.326645**, LR **9.293e-4**, EMA **0.996851**, elapsed **50,881 s**, ETA **188,207 s (~52h17m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti use **6,986/16,303 MiB** at **19%**. The ViT resume and evaluation queues remain gated; seed-0 downstream result count is **0/8**.

Epoch-51 fixed validation by source was NRPos Indoor **0.509978**, RFP **0.138714**, spectrogram **0.186611**, and WiFi CSI **0.471279**. Relative to epoch 50, NRPos improved slightly while the other three rose; the aggregate loss also rose from **0.324860** to **0.326645**. This was not a new best and does not change checkpoint selection.

Refreshed the comparison summaries and PNGs under `../runs/latent-wave/comparison/current/` through completed epoch 51 and updated `figure_index.md`. Commands used are the same four run-summary/plot commands recorded in the epoch-50 entry and were rerun successfully against the same seed-0 run directories. Paper reference values remain separate from local downstream status.

### Mamba live status and recovery command — 2026-10-08 05:01 MDT

Mamba-2 seed 0 PID **1149902** remains active at epoch **52/240**, step **10,950/51,360** (36/214 in this epoch). Rolling mean train loss is **0.313095**; latest completed fixed validation is **0.326645** from epoch 51; best remains epoch 50 at **0.324860**. LR is **9.290e-4**, EMA **0.996853**, elapsed **50,978 s**, ETA **188,131 s (~52h15m)**, PyTorch memory **2.28/5.12 GiB**, RTX 5070 Ti use **6,995/16,303 MiB** at **15%** utilization. The latest checkpoint remains completed epoch 51/global step 10,914. ViT handoff and downstream queues remain gated; no downstream results exist.

If this trainer exits before 240 epochs, use the command below **only after confirming PID 1149902 has exited**. It resumes from the current run's latest checkpoint and preserves its best-checkpoint selection. Do not start it while the existing trainer is active.

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

### Mamba in-epoch poll — 2026-10-08 05:04 MDT

Mamba-2 seed 0 PID **1149902** remains active at epoch **52/240**, step **10,990/51,360** (76/214 in the epoch). Rolling mean train loss is **0.313677**; latest completed fixed validation is **0.326645** from epoch 51; best remains epoch 50 at **0.324860**. LR is **9.283e-4**, EMA **0.996856**, elapsed time **51,174 s**, ETA **187,979 s (~52h13m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti use **6,992/16,303 MiB** at **14%** utilization. The latest checkpoint remains epoch 51/global step 10,914. All queue processes remain gated and downstream outputs remain **0/8**. The figures remain current through completed epoch 51.

### Mamba epoch 52 and current results snapshot — 2026-10-08 05:15–05:18 MDT

Mamba-2 seed 0 completed **52/240** epochs at global step **11,128**. Mean training JEPA loss was **0.315489**, fixed-protocol validation loss was **0.334982**, epoch duration was **1,051.969 s**, and throughput was **52.078 source examples/s**. Validation increased from **0.326645** at epoch 51; the best checkpoint remains epoch **50** at **0.324860**. Run metadata records Mamba-2 seed 0, 52 completed epochs, global step 11,128, the four configured pretraining datasets, 19,085,440 total parameters, CUDA `cuda:0`, and the RTX 5070 Ti. A read-only CPU audit confirms that `latest.pt` stores Mamba-2/seed 0, zero-based epoch 51 (52 completed), global step 11,128, best zero-based epoch 49, and the fixed validation protocol. All 19,085,440 model tensor values are finite; optimizer, both scheduler, sampler, RNG, and resolved-config state needed for resume are present.

At **05:18 MDT**, PID **1149902** was active at epoch **53/240**, step **11,150/51,360** (22/214), with rolling train loss **0.311476**, latest completed validation **0.334982**, LR **9.2566e-4**, EMA **0.996868**, elapsed time **51,964 s**, ETA **187,397 s (~52h03m)**, PyTorch memory **2.28/5.12 GiB**, and RTX 5070 Ti use **6,988/16,303 MiB** at **18%**. ViT seed 0 remains at one completed epoch. The guarded ViT handoff and all downstream/evaluation queues are still waiting for full Mamba seed-0 completion; no downstream result JSONs exist and the local downstream status remains not run.

The current PNGs and CSV/JSON summaries were refreshed through completed Mamba epoch 52 under `../runs/latent-wave/comparison/current/`; `figure_index.md` now describes that snapshot. Paper Table II scores remain clearly marked as paper references. The local Table III ablation remains unrun. These pretraining validation changes do not establish downstream improvement or a Mamba-versus-ViT result.

### Mamba live poll — 2026-10-08 15:02 MDT

PID **1149902** remains active at epoch **86/240**, step **18,240/51,360** (**50/214**). The rolling mean train loss is **0.300886**; latest completed fixed-validation loss remains **0.325219** from epoch 85. LR is **7.6655e-4**, EMA **0.997421**, elapsed **86,402 s**, ETA **156,888 s (~43h35m)**, PyTorch memory **2.42/6.14 GiB**, and RTX 5070 Ti use **7,959/16,303 MiB** at **16%** utilization. Metadata/checkpoint remain at epoch 85, global step **18,190**, best epoch 72. The seed-0 evaluator, ViT handoff, matched-seed queue, and fine-tuning queue are still gated; no downstream outputs exist. PNGs remain accurate through epoch 85 because epoch 86 has not completed.

### Mamba live poll — 2026-10-08 15:04 MDT

PID **1149902** remains active at epoch **86/240**, step **18,260/51,360** (**70/214**). Rolling mean train loss is **0.299888**; latest completed fixed-validation loss remains **0.325219** from epoch 85. LR is **7.6601e-4**, EMA **0.997422**, elapsed **86,492 s**, ETA **156,785 s (~43h33m)**, PyTorch memory **2.42/6.14 GiB**, and RTX 5070 Ti use **7,960/16,303 MiB** at **13%** utilization. Metadata/checkpoint remain at epoch 85, global step **18,190**, best epoch 72. The evaluator and queued ViT, matched-seed, and fine-tuning runs remain gated; no downstream outputs exist. No newer completed epoch is available for plotting.

### Mamba epoch 87 complete; plots refreshed — 2026-10-08 15:33–15:35 MDT

Mamba-2 seed 0 completed epoch **87/240** at global step **18,618**. Mean training JEPA loss was **0.298700236** and fixed-protocol validation loss was **0.324806435**. Relative to epoch 86, training loss fell **0.53%** while validation rose **0.90%**. The best fixed-validation checkpoint remains epoch **72** at **0.320415764**; epoch 87 is **1.37%** above it. Per-source validation was spectrogram **0.211814** (**+3.39%**), RFP **0.140118** (**+1.04%**), WiFi CSI **0.478139** (**+0.93%**), and NRPos Indoor **0.469155** (**−0.27%**) versus epoch 86. Epoch duration was **1,028.804 s** at **53.250 source examples/s**. From epoch 20 to 87, mean train and fixed-validation losses fell **13.28%** and **7.21%**; the mean validation loss over epochs 70–87 is **0.326524**. This remains pretraining-only evidence and does not show downstream improvement.

Run metadata and `latest.pt` confirm Mamba-2/seed 0, **87** completed epochs/global step **18,618**, last validation **0.324806435**, and best epoch **72**. The latest checkpoint was written at **15:33:49 MDT**. At **15:35 MDT**, PID **1149902** was active at epoch **88/240**, step **18,630/51,360** (**12/214**), rolling train loss **0.303851**, latest completed validation **0.324806**, LR **7.5585e-4**, EMA **0.997451**, elapsed **88,293 s**, ETA **155,118 s (~43h05m)**, PyTorch allocation/reservation **2.42/6.14 GiB**, and RTX 5070 Ti use **7,958/16,303 MiB** at **15%** utilization. The evaluator and ViT/matched-seed/fine-tuning queues remain gated; downstream result count remains **0/8**.

Reran the same CLI-verified summary and plot commands recorded for epoch 86, writing to `../runs/latent-wave/comparison/current/`. The training/validation histories, per-source curves, compute comparison, downstream status, and Table II paper-reference comparison now include epoch 87 and report Mamba **87/240**, ViT **1/240**. Visual inspection confirmed that legacy and fixed validation remain separately labeled, paper values remain reference-only, and no local downstream score is plotted. The plotter collapsed one exact copied epoch record. The Table III plot remains a paper-reference-only figure.

### Mamba live poll — 2026-10-08 15:37 MDT

PID **1149902** remains active at epoch **88/240**, step **18,650/51,360** (**32/214**). Rolling train loss is **0.302714**; latest completed fixed-validation loss remains **0.324806** from epoch 87. LR is **7.5530e-4**, EMA **0.997452**, elapsed **88,384 s**, ETA **155,016 s (~43h03m)**, PyTorch memory **2.42/6.14 GiB**, and RTX 5070 Ti use **7,944/16,303 MiB** at **18%** utilization. Metadata/checkpoint remain at epoch 87, step **18,618**, best epoch 72. The evaluator and queued ViT, matched-seed, and fine-tuning runs remain gated; downstream result count remains **0/8**. PNGs remain current through completed epoch 87.

### Mamba epoch 86 complete; plots refreshed — 2026-10-08 15:16–15:18 MDT

Mamba-2 seed 0 completed epoch **86/240** at global step **18,404**. Mean training JEPA loss was **0.300285000** and fixed-protocol validation loss was **0.321914898**. Relative to epoch 85, training loss fell **1.23%** and validation loss fell **1.02%**. Epoch 86 remains **0.47%** above the best fixed-validation checkpoint at epoch **72** (**0.320415764**). Per-source validation at epoch 86 was spectrogram **0.204868**, RFP **0.138678**, WiFi CSI **0.473710**, and NRPos Indoor **0.470405**; relative to epoch 85, these changed by **−3.34%**, **−3.76%**, **+0.51%**, and **−0.66%**, respectively. Epoch duration was **1,027.281 s** at **53.329 source examples/s**. From epoch 20 to 86, aggregate training and validation losses fell **12.82%** and **8.03%**; the mean fixed-validation loss over epochs 70–86 is **0.326625**. This improves pretraining validation loss but does not establish downstream improvement.

Run metadata and `latest.pt` confirm Mamba-2/seed 0, **86** completed epochs, global step **18,404**, fixed validation, and best epoch **72**; both were written at **15:16 MDT**. The same foreground trainer PID **1149902** advanced into epoch 87. At **15:17 MDT**, it was at step **18,420/51,360** (**16/214**), rolling train loss **0.297184**, latest completed validation **0.321915**, LR **7.6164e-4**, EMA **0.997435**, elapsed **87,284 s**, ETA **156,088 s (~43h21m)**, PyTorch allocation/reservation **2.42/6.14 GiB**, and RTX 5070 Ti memory use **7,946/16,303 MiB** at **17%** utilization. The gated evaluator and later-run queues remain waiting; seed-0 downstream results remain **0/8**.

Regenerated summary tables and plots under `../runs/latent-wave/comparison/current/` using the CLI-verified `summarize_training_runs.py`, `plot_training_history.py`, `plot_compute_comparison.py`, and `plot_paper_comparison.py` commands against the original and fixed-validation seed-0 Mamba/ViT run directories. The PNG curves now include epoch 86; the paper comparison reports Mamba **86/240**, ViT **1/240**, and labels all published Table II bars as paper references. The downstream-status plot still reports all four tasks not evaluated. Visually inspected the validation curve, protocol panels, per-source histories, and paper comparison; the different legacy/fixed validation protocols remain distinguished, and no local downstream score is plotted. The plotter collapsed one exact copied epoch record. Table III remains a paper-reference-only plot; no local masking ablation was run.

### Training diagnosis and resume — 2026-10-09

The apparent lack of improvement is not supported by the aggregate fixed-validation history. From epoch 20 to 118, mean training JEPA loss fell **19.41%** (0.344455 to 0.277609) and fixed-validation loss fell **11.10%** (0.350029 to 0.311163); the best remains epoch **115** at **0.308417**. The epoch-to-epoch validation curve is noisy, and the current plot separates the legacy epoch 1–19 protocol from the fixed protocol beginning at epoch 20. Those early legacy values cannot be compared directly to the later fixed-validation values.

There is a real per-source generalization gap to monitor: from epoch 20 to 118, spectrogram training loss fell **12.66%** while spectrogram validation rose **24.82%**. Validation fell **9.46%** for RFP, **10.44%** for WiFi CSI, and **24.30%** for NRPos Indoor. Code inspection found no confirmed loss, mask, target-EMA, gradient, or optimizer defect; therefore the controlled prose config remains unchanged, and no downstream test data informed this diagnosis.

The graceful stop completed at epoch **119/240**, global step **25,466**. Mean train loss was **0.275280154** and fixed-validation loss **0.309019295**, both lower than epoch 118; best remains epoch **115** at **0.308417416**. The latest checkpoint was verified to include the model, optimizer, schedulers, sampler/RNG states, and resolved config. The foreground wrapper resumed this same run with the original prose config and append-only log; resume preflight returned `(48, 28, 256)` on CUDA and loaded zero-based checkpoint epoch 118/global step **25,466**. At the latest poll, PID **1870291** was active at epoch **120/240**, step **25,490/51,360** (**24/214**), rolling train loss **0.271529**, latest validation **0.309019**, GPU memory **3,941/16,303 MiB**, and ETA about **34h16m**. The ViT and evaluation queues remain gated, with downstream results still **0/8**.

At **01:20 MDT**, the resumed Mamba trainer PID **1870291** was active at epoch **120/240**, step **25,510/51,360** (**44/214**), rolling train loss **0.271881**, latest completed validation **0.309019**, LR **5.47195e-4**, EMA **0.997987**, GPU memory **4,324/16,303 MiB**, and ETA about **34h14m**. The other GPU queues remain gated and downstream scores remain unavailable.

At **01:21 MDT**, the same trainer advanced to step **25,520/51,360** (**54/214**), with rolling train loss **0.271250**, latest completed validation **0.309019**, GPU memory **4,770/16,303 MiB**, and ETA about **34h14m**. Metadata and checkpoint remain at the last completed epoch 119 until epoch 120 completes; all downstream score files remain absent.

Checkpoint audit confirms Mamba `latest.pt` is architecture `mamba2`, seed **0**, 119 completed epochs/global step **25,466**, best epoch **115** (validation **0.308417416**), with optimizer/scheduler/sampler/RNG/resolved-config resume state. The ViT seed-0 logs explicitly show a SIGTERM request and a checkpointed stop after epoch 1; no exception is present, so the queued resume is a continuation rather than crash recovery. At **01:22 MDT**, Mamba PID **1870291** was at epoch **120**, step **25,530/51,360** (**64/214**), rolling train loss **0.270743**, latest validation **0.309019**, RTX 5070 Ti use **4,768/16,303 MiB**, ETA about **34h13m**. Downstream results remain **0/8**.

At **01:23 MDT**, PID **1870291** advanced to step **25,550/51,360** (**84/214**), rolling train loss **0.270675**, latest completed validation **0.309019**, GPU memory **4,762/16,303 MiB**, and ETA about **34h12m**. The current epoch remains incomplete; downstream result count is still **0/8**.

### Mamba epoch 120 complete — 2026-10-09 01:34 MDT

Mamba-2 seed 0 completed epoch **120/240** at global step **25,680**. Mean training loss was **0.270592213** and fixed-validation loss **0.308937338**. Relative to epoch 119, train loss fell **1.70%** and validation loss fell **0.03%**; best remains epoch **115** at **0.308417416**. From fixed-validation epoch 20, aggregate train and validation losses are **21.44%** and **11.74% lower**. Per-source validation losses were spectrogram **0.235712**, RFP **0.124352**, WiFi CSI **0.453582**, and NRPos Indoor **0.422103**. Epoch duration was **1,129.729 s**, throughput **48.493 source examples/s**.

Metadata and `latest.pt` verify 120 completed epochs, step **25,680**, fixed-validation protocol, best epoch 115, and resume state. At **01:35 MDT**, PID **1870291** was active in epoch **121**, step **25,690/51,360**, rolling train loss **0.263105**, latest validation **0.308937**, GPU memory approximately **4.76 GiB**, and ETA about **34h02m**. Refreshed current PNG/CSV/JSON/Markdown outputs include completed epoch 120 only; validation panels preserve the legacy/fixed protocol distinction. ViT remains at 1/240; downstream JSON result count remains **0/8**.


## Prior run-status snapshot — 2026-10-09 01:38 MDT

Mamba-2 seed 0 has **120/240** completed epochs and is actively training epoch **121**, step **25,720/51,360** (**40/214** for the epoch). The rolling train loss is **0.268224582**; latest completed fixed-validation loss is **0.308937338** (epoch 120); best validation remains **0.308417416** at epoch 115. The LR is **5.40466e-4**, EMA momentum **0.998003**, elapsed time **122,761 s**, and ETA approximately **34h**. PyTorch allocation/reservation is **2.09/2.94 GiB**; the RTX 5070 Ti reports **4,763/16,303 MiB** at **17%** utilization. The log advanced ten steps in a 15-second observation, confirming the process is progressing.

The fixed-validation comparison from epoch 20 to 120 shows aggregate training loss down **21.44%** and validation down **11.74%**. Validation fell **4.51%** from epoch 111 to 120, although the final value is **0.17%** above the best at epoch 115. Per-source behavior explains why aggregate progress can look uneven: from epoch 20 to 120, spectrogram training loss fell **17.54%** while spectrogram validation rose **23.56%**; validation fell **6.13%** on RFP, **11.94%** on WiFi CSI, and **24.87%** on NRPos Indoor. This is evidence of a source-specific validation gap, not proof of a broken training implementation. No confirmed code defect has been found, so the active controlled configuration is unchanged.

ViT seed 0 remains at its preserved epoch-1 checkpoint; its guarded resume waits for Mamba to confirm all 240 epochs. The GPU remains sequentially allocated, and local downstream results remain **0/8**. The current PNG files use completed epochs only and therefore end at Mamba epoch 120.


### Mamba live poll — 2026-10-09 01:47 MDT

- PID **1870291** remains active in epoch **121/240**, at global step **25,830/51,360** (**150/214** optimizer steps in this epoch). Rolling mean train loss is **0.270066827**; latest completed fixed validation is **0.308937338** (epoch 120), and selected best remains epoch **115** (**0.308417416**). LR is **5.36939e-4**, EMA **0.998012**, elapsed **123,315 s**, and ETA **121,883 s (~33h51m)**. PyTorch allocation/reservation is **2.09/3.39 GiB**; RTX 5070 Ti use is **5,232/16,303 MiB** at **16%**.
- The run remains active and is the only GPU compute job. Metadata/checkpoints correctly remain at 120 completed epochs while epoch 121 is incomplete. The guarded ViT and linear-evaluation queues continue waiting for full Mamba completion; the results directory contains only `evaluation_console.log`, not score JSONs.
- A read-only `src/evaluate_linear.py --help` probe stalled in filesystem I/O with no output while training was active. I sent SIGTERM to that help-only process; it exited, launched no evaluation, and did not touch the trainer or checkpoints. Existing documented task identifiers and evaluator protocol remain in force.


## Current run status — 2026-10-09 01:53 MDT

Mamba-2 seed 0 has **121/240** completed epochs and is active in epoch **122**, step **26,000/51,360** (**106/214**). The rolling train loss is **0.271211305**; latest completed fixed-validation loss is **0.311959330** from epoch 121; the best checkpoint remains epoch 115 at **0.308417416**. LR is **5.31484e-4**, EMA **0.998025**, elapsed **124,211 s**, and ETA approximately **33h39m**. PyTorch allocation/reservation is **2.09/3.39 GiB**; the RTX 5070 Ti reports **5,230/16,303 MiB** at **14%** utilization.

Epoch 121 train loss fell **0.17%** from epoch 120 while validation rose **0.98%**, so the best checkpoint did not change. Fixed-validation loss from epoch 20 to 121 is down **21.58%** for training and **10.88%** for validation. Per-source validation was mixed at epoch 121: spectrogram **+5.37%**, RFP **+4.65%**, WiFi CSI **−0.94%**, and NRPos Indoor **−0.49%** versus epoch 120. This update is label-free validation evidence only; no downstream result has been used for selection.

The refreshed PNGs and machine-readable summaries include completed Mamba epoch 121 and exclude incomplete epoch 122. ViT seed 0 remains at **1/240**, its guarded continuation waits for Mamba to confirm 240 epochs, and downstream evaluation has not started (**0/8** result files).


## Current snapshot — 2026-10-09 03:46 MDT

Mamba-2 seed 0 completed **127/240** epochs and is active in epoch **128**, at global step **27,200/51,360** (**22/214** optimizer steps). Epoch 127 training JEPA loss was **0.270670247** and fixed-validation loss **0.303631013**, a new best by **0.000105478** over epoch 124. From epoch 126, training loss increased **0.49%** while validation loss decreased **2.66%**. The current rolling train loss is **0.266451**; latest completed validation remains **0.303631013**. LR is **4.92910e-4**, EMA **0.998118**, elapsed training time **130,470 s**, and ETA about **115,888 s (~32h11m)**. PyTorch allocated/reserved memory is **2.23/4.94 GiB**; the RTX 5070 Ti is at **6,809/16,303 MiB** and **16%** utilization.

The PNG snapshot refreshed at 03:45 MDT plots completed epochs only: Mamba through epoch 127 and ViT through epoch 1. Downstream evaluation has not run and the four local score panels remain unevaluated (**0/8** result files). The paper comparison chart includes published reference results but does not imply local scores. ViT’s guarded continuation still waits for Mamba’s full run; no second GPU job is running.


## Diagnosis and resumed run — 2026-10-09 04:09 MDT

On the user's request, Mamba seed 0 was stopped cleanly at the epoch boundary. Epoch 128 completed at global step **27,392** with train loss **0.266441049** and validation loss **0.305403809**. The best checkpoint remains epoch **127** at **0.303631013**. The exact checkpoint includes optimizer/scheduler state and RNG state; the run resumed from `latest.pt` using the same prose config, seed, output directory, and fixed-validation protocol. CUDA preflight passed with output shape `(48, 28, 256)`.

The fixed-validation aggregate improved over epochs 20–127: train loss declined **21.42%**, validation declined **13.26%**, OLS slope **−0.00031442 per epoch**. There is a source-specific validation gap on spectrogram (training **−14.00%**, validation **+25.50%**); the other three source validation losses declined. A reconstructed train/validation split has similar simple per-image value statistics, and the trained target features retain nonzero spread and effective rank across sources. A real-batch gradient check was finite/nonzero for student and predictor and absent for the frozen target, as intended. No optimizer, EMA, masking/alignment, or representation-collapse defect was found. The spectrogram behavior may reflect source-specific overfitting, but these measurements do not prove its cause; the primary controlled config was therefore left unchanged.

The read-only diagnostics are recorded in `../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val/diagnostic_20261009.log`. The Mamba run is active again at **128 completed epochs**, best epoch **127**; ViT remains **1/240** and downstream results remain **0/8**. The current comparison is still partial.

The original ViT seed-0 log shows a handled SIGTERM after epoch 1 with no exception. Its source checkpoint matches the fixed-validation resume copy byte-for-byte. A guarded handoff is active and will resume ViT from that checkpoint with the fixed-validation selection reset after Mamba finishes; the seed-0 evaluation queue is also active and still blocks all downstream test evaluation until both 240-epoch runs finish.


## Post-resume progress — 2026-10-09 04:28 MDT

Mamba seed 0 completed epoch **129** after resuming from its epoch-128 checkpoint. The epoch-129 checkpoint is verified; global step **27,606**, train loss **0.268822671**, fixed-validation loss **0.299383451**, and new selected best at epoch **129**. Validation fell **1.97%** from epoch 128; all four per-source validation losses are below their epoch-127 values. This is measured improvement on the fixed pretraining validation protocol, not downstream evidence.

The loss, source, compute, validation-protocol, downstream-status, and paper-reference PNGs were refreshed through completed epoch **129** at 04:28 MDT. At the latest poll Mamba was active in epoch **130**, step **27,610/51,360**, rolling loss **0.262934**, LR **4.79730e-4**, EMA **0.998150**, RTX 5070 Ti use **4,605/16,303 MiB**, and ETA approximately **31h41m**. ViT is still **1/240** pending the guarded handoff; no downstream results exist (**0/8**).


## Current snapshot — 2026-10-09 04:47 MDT

Mamba seed 0 completed **130/240** epochs. Epoch 130 training loss was **0.263218057** and fixed-validation loss **0.303166182**; the selected best remains epoch **129** at **0.299383451**. Training loss declined while validation rebounded modestly. The current active epoch is **131**, step **27,830/51,360** (**10/214**), rolling train loss **0.264864**, latest validation **0.303166182**, LR **4.72663e-4**, EMA **0.998167**, elapsed time **133,740 s**, and ETA about **113,076 s (~31h24m)**. RTX 5070 Ti use is **5,051/16,303 MiB** at **16%** utilization.

The PNG snapshot refreshed at 04:47 MDT includes completed Mamba epochs through 130 and ViT epoch 1. ViT remains gated behind Mamba; seed-0 downstream evaluation and the matched-seed queue are also gated. Local downstream results remain **0/8**, so there is no Mamba-versus-ViT downstream conclusion yet.

## Live snapshot — 2026-10-09 04:50 MDT

Mamba seed 0 is active at epoch **131**, step **27,870/51,360** (**50/214**), with rolling train loss **0.266626405**. Most recently completed validation is **0.303166182** (epoch 130); current selected checkpoint remains epoch **129** at **0.299383451**. LR is **4.71378e-4**, EMA **0.998171**, logged elapsed **133,949 s**, and ETA **112,897 s (~31h22m)**. The live trainer reports **2.03/3.22 GiB** PyTorch allocated/reserved; total RTX 5070 Ti use was **5,051/16,303 MiB** at **16–17%** utilization. The epoch-130 checkpoint remains the latest periodic checkpoint. ViT is still **1/240** and downstream results **0/8**; the existing handoff/evaluation queues remain gated. PNGs remain current through completed Mamba epoch 130 and will be refreshed after the next epoch completes.


## Epoch 131 and refreshed figures — 2026-10-09 05:05 MDT

Mamba seed 0 completed epoch **131/240** at global step **28,034**. Mean train JEPA loss was **0.263676098** and fixed-validation loss **0.299571795**; best remains epoch **129** at **0.299383451**. Epoch duration was **1,124.825 s** and throughput **48.704 source examples/s**. The latest checkpoint and metadata were verified on disk. The live run is in epoch **132**, step **6/214**, with rolling train loss **0.259925**, most recent validation **0.299571795**, LR **4.65922e-4**, EMA **0.998184**, ETA about **31h09m**, and total GPU use **5,519/16,303 MiB**. ViT remains stopped after 1/240 and queued behind Mamba; downstream results remain **0/8**.

Refreshed figures and machine-readable summaries at **05:05 MDT** through completed Mamba epoch 131 and ViT epoch 1. See `../runs/latent-wave/comparison/current/figure_index.md`; the loss curves exclude the incomplete epoch 132. The four downstream result slots remain explicitly not evaluated, and the Table II bars are paper-reported reference values only.


### Throughput note (provisional)

A read-only sample at 05:06 MDT found the HDF5 inputs on the Windows-backed `/mnt/c` `9p` mount, one CPU core busy in the Mamba parent, DataLoader workers mostly sleeping, and GPU utilization at **16–17%**. Mamba epoch 131 measured **48.704 source examples/s**; the earlier ViT epoch measured **196.643 source examples/s**. The trainer is configured for four pinned-memory loader workers, and Mamba uses bidirectional official SSM blocks. These observations show a runtime-throughput gap and make storage or CPU-side overhead plausible, but do not isolate its cause; the single ViT epoch is only a rough speed reference. No active run was altered.


## Epoch 132 and refreshed figures — 2026-10-09 05:24 MDT

Mamba seed 0 completed epoch **132/240** at global step **28,248**. Mean train JEPA loss was **0.261632370**; fixed-validation loss was **0.296659710**, a new best (0.91% below the previous best, epoch 129 at **0.299383451**). Epoch duration was **1,126.519 s** at **48.631 source examples/s**. The checkpoint and metadata were verified on disk. The live trainer is in epoch **133**, step **2/214**, with rolling loss **0.259232**, latest validation **0.296659710**, ETA about **30h54m**, and total GPU use **5,515/16,303 MiB**. ViT remains stopped at **1/240** pending the guarded handoff; downstream results remain **0/8**.

Refreshed all training-history, validation, source-loss, compute, status, and paper-comparison PNGs and CSV/JSON summaries at **05:24 MDT** through completed epoch 132. The figures continue to label validation protocols separately, identify paper values as references, and show downstream scores as unavailable. See `../runs/latent-wave/comparison/current/figure_index.md`.


## Epoch 133 and refreshed figures — 2026-10-09 05:42 MDT

Mamba seed 0 completed epoch **133/240** at global step **28,462**. Mean train JEPA loss was **0.260605378**, fixed-validation loss **0.298501656**. The validation loss rose **0.62%** from epoch 132, so the selected best remains epoch **132** at **0.296659710**. Epoch duration was **1,100.743 s**, throughput **49.770 source examples/s**. The checkpoint and metadata were verified. Mamba is now in epoch **134**, step **8/214**, rolling loss **0.257749**, latest validation **0.298501656**, with ETA about **30h36m** and total GPU use **5,515/16,303 MiB**. ViT remains stopped at **1/240** pending its guarded handoff; local downstream scores remain **0/8**.

The training-history, validation, source-loss, compute, downstream-status, and paper-comparison PNGs were refreshed through completed epoch 133. They label the differing validation protocols and paper reference values, and show downstream status as not evaluated. The updated inventory is `../runs/latent-wave/comparison/current/figure_index.md`.


## Live progress — 2026-10-09 05:44 MDT

Mamba seed 0 is active in epoch **134**, step **30/214**, with rolling loss **0.2601714456**, latest validation **0.2985016557**, and ETA about **30h35m**. GPU use was **5,515/16,303 MiB** at **16%**. The latest completed checkpoint remains epoch **133** and the selected best remains epoch **132** at **0.296659710**. ViT is still stopped at **1/240** behind the guarded handoff; local downstream results remain **0/8**. PNGs are current through completed epoch 133.


At **05:46 MDT**, Mamba seed 0 is active at epoch **134**, step **50/214**, rolling loss **0.260172**, latest validation **0.298502**, best validation **0.296660** at epoch 132, ETA about **30h34m**. RTX 5070 Ti use is **5,515/16,303 MiB** at **17%**. ViT remains at **1/240**; downstream results remain **0/8**.


At **05:48 MDT**, Mamba seed 0 is active in epoch **134**, step **70/214**, rolling loss **0.259241**, latest validation **0.298502**, and best validation **0.296660** at epoch 132. ETA is about **30h32m**; GPU use was **5,513/16,303 MiB** at **19%**. ViT remains **1/240**, and local downstream result files remain **0/8**. The current PNGs show completed epochs only through Mamba 133.


## Epoch 134 and refreshed figures — 2026-10-09 06:01 MDT

Mamba seed 0 completed epoch **134/240** at global step **28,676**. Mean train loss was **0.258628203**, fixed-validation loss **0.297035951**. The selected best remains epoch **132** at **0.296659710**. Epoch duration was **1,115.815 s**, throughput **49.098 source examples/s**. The checkpoint and metadata were verified. Mamba is now in epoch **135**, step **4/214**, rolling loss **0.257405**, latest validation **0.297035951**, ETA about **30h21m**, and GPU use **5,517/16,303 MiB**. ViT remains **1/240**, queued behind Mamba; local downstream results remain **0/8**.

Refreshed all loss/compute/status and Table II PNGs plus machine-readable summaries through completed epoch 134. See `../runs/latent-wave/comparison/current/figure_index.md`. Validation protocols remain separately labeled, and paper values are reference-only.


## Epoch 135 and refreshed figures — 2026-10-09 06:20 MDT

Mamba seed 0 completed epoch **135/240** at global step **28,890**. Mean train loss was **0.258994895**, and fixed-validation loss was **0.297408744**. Best remains epoch **132** at **0.296659710**. Epoch duration was **1,121.447 s**, throughput **48.851 source examples/s**. The latest checkpoint was verified. Mamba is in epoch **136**, step **10/214**, rolling loss **0.259089**, latest validation **0.297409**, ETA about **30h04m**, with GPU use **6,177/16,303 MiB**. ViT remains **1/240** behind the sequential handoff; downstream results remain **0/8**.

Refreshed the training-history, validation, source-loss, compute, task-status, and paper-comparison figures through Mamba epoch 135. The figure index and machine-readable summaries were updated. Local downstream results are still unavailable; published values remain labeled as references.


At **06:23 MDT**, Mamba seed 0 was active in epoch **136**, step **40/214**, with rolling train loss **0.264274**, latest validation **0.297409**, best validation **0.296660** at epoch 132, and ETA about **30h02m**. RTX 5070 Ti use was **6,177/16,303 MiB** at **18%**. ViT remains at **1/240**; downstream results remain **0/8**. Figures are current through completed Mamba epoch 135.


At **06:24 MDT**, Mamba seed 0 was active in epoch **136**, step **60/214**, rolling loss **0.265141**, latest validation **0.297409**, selected best **0.296660** at epoch 132, and ETA about **30h01m**. GPU use was **6,177/16,303 MiB** at **15%**. ViT remains at **1/240**; downstream results remain **0/8**. Figures are current through completed epoch 135.


At **06:26 MDT**, Mamba seed 0 was in epoch **136**, step **80/214**, rolling loss **0.265256**, latest validation **0.297409**, and best **0.296660** at epoch 132. ETA is about **29h59m**; GPU use was **6,177/16,303 MiB** at **15%**. ViT remains **1/240**, and downstream results remain **0/8**. Figures are current through completed epoch 135.


At **06:27 MDT**, Mamba seed 0 was active in epoch **136**, step **90/214**, rolling loss **0.264822**, latest validation **0.297409**, best validation **0.296660** at epoch 132, ETA about **29h58m**. GPU use was **6,177/16,303 MiB** at **15%**. ViT remains at **1/240**, and local downstream results remain **0/8**. Figures are current through completed epoch 135.


## Live progress — 2026-10-09 06:30 MDT

Mamba-2 seed 0 remains active as PID **1941478** in epoch **136/240**, step **29,020/51,360** (**130/214** optimizer steps in this epoch). Rolling mean train JEPA loss is **0.263897436**; latest completed validation is **0.297408744** (epoch 135), and the selected best remains **0.296659710** (epoch 132). LR is **4.34569e-4**, EMA momentum **0.998260**, elapsed **139,951 s**, ETA **107,736 s (~29h56m)**. The RTX 5070 Ti reports **6,177/16,303 MiB** used at **16%** utilization; PyTorch reports **2.26/4.32 GiB** allocated/reserved. The epoch-135 checkpoint is the latest completed-epoch checkpoint. ViT remains at **1/240** behind the sequential handoff; local downstream results remain **0/8**. Current PNGs show completed epochs only through Mamba epoch 135.


## Live progress — 2026-10-09 06:32 MDT

Mamba-2 seed 0 remains active in epoch **136/240**, step **29,040/51,360** (**150/214** optimizer steps). Rolling mean JEPA loss is **0.263306968**; the latest completed validation remains **0.297408744** (epoch 135), with best **0.296659710** at epoch 132. LR is **4.33931e-4**, EMA momentum **0.998262**, elapsed **140,053 s**, ETA **107,644 s (~29h54m)**. RTX 5070 Ti use is **6,177/16,303 MiB** at **19%** utilization; PyTorch reports **2.26/4.32 GiB** allocated/reserved. The latest completed checkpoint remains epoch 135. ViT is queued at epoch 1/240, and downstream evaluation remains gated; no local downstream scores are available. Figures remain current through completed Mamba epoch 135.


## Epoch 136 and refreshed figures — 2026-10-09 06:39 MDT

Mamba-2 seed 0 completed epoch **136/240** at global step **29,104**. Mean train JEPA loss was **0.262265141** and fixed-protocol validation loss was **0.295680759**, a new best (previous best: epoch 132, **0.296659710**). Epoch duration was **1,125.649 s**, throughput **48.669 source examples/s**. The latest and best checkpoints were saved at **06:37:57 MDT**; metadata reports 136 completed epochs and the fixed validation protocol.

At the **06:39 MDT** poll, Mamba was active in epoch **137**, step **6/214**, rolling loss **0.258908**, latest validation **0.295681**, LR **4.31701e-4**, EMA **0.998267**, ETA about **29h49m**, and GPU use **6,173/16,303 MiB** (**15%**). ViT remains **1/240**, queued behind Mamba; local downstream results remain **0/8**.

Refreshed the training-history, validation, source-loss, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 136. See `../runs/latent-wave/comparison/current/figure_index.md`. Paper values are labeled as references; no local downstream scores are available.

Fixed-protocol trend check: comparing Mamba seed 0 epoch 20 with epoch 136 in `pretraining_loss_traces.csv`, mean training JEPA loss decreased **23.86%** (0.34446 to 0.26227) and mean validation JEPA loss decreased **15.53%** (0.35003 to 0.29568). Both rows use `fixed_seed_masks_channels_each_example_once_v1`. This is evidence of lower held-out JEPA loss during pretraining; it does not establish downstream superiority over ViT.


## Source-level validation diagnostic — 2026-10-09 06:43 MDT

Compared fixed-protocol Mamba seed-0 epochs 20 and 136 from the generated per-source trace:

| Pretraining source | Train-loss change | Validation-loss change |
|---|---:|---:|
| Spectrogram | −20.60% | **+24.13%** |
| RFP | −15.91% | −14.28% |
| WiFi CSI | −22.48% | −15.96% |
| NRPos indoor | −28.15% | −28.89% |

Code inspection found that `src/pretrain_jepa.py::_build_loaders` makes a deterministic, unlabeled 95/5 sample split independently within each configured source. `_evaluate` fixes the validation RNG and mask/channel sampling; `src/datasets/pretrain.py` reads the configured cached tensor without another source-specific transform. The inventory says the configured spectrogram cache is already transformed/normalized and has no labels or group IDs. This provides no evidence of a validation-preprocessing mismatch or confirmed implementation bug. The spectrogram train/validation divergence is consistent with a source-specific generalization gap or sample heterogeneity; its cause is unresolved. The active primary run is unchanged. Consider only bounded, single-factor validation tuning after the matched primary runs release the GPU; do not select via downstream test results.

At the **06:43 MDT** live poll, Mamba-2 seed 0 remained active in epoch **137/240**, step **29,160/51,360** (**60/214**). Rolling train loss was **0.258933**; latest validation was **0.295681** (epoch 136, current best), LR **4.30108e-4**, EMA **0.998271**, ETA **107,119 s (~29h45m)**. RTX 5070 Ti use was **6,173/16,303 MiB** at **16%**. ViT and downstream queues remain gated; no downstream score is available.


## Live progress — 2026-10-09 06:45 MDT

Mamba-2 seed 0 is active in epoch **137/240**, step **29180/51360** (**76/214** in this epoch). Rolling mean train loss is **0.259572857**; latest completed validation is **0.295680759** (epoch 136, current best). LR **0.00042947162**, EMA **0.998272586**, elapsed **140803 s**, ETA **107026 s (~29.7 h)**. PyTorch allocation/reservation is **2.26/4.32 GiB**; latest `nvidia-smi` poll reports **6,173/16,303 MiB** at **19%** on the RTX 5070 Ti. ViT remains queued at epoch 1; downstream evaluations have not started.


## Epoch 137 and refreshed figures — 2026-10-09 06:57 MDT

Mamba-2 seed 0 completed epoch **137/240** at global step **29,318**. Mean train JEPA loss was **0.259838584** and fixed-protocol validation loss was **0.298607979**. The selected best remains epoch **136** at **0.295680759**. Epoch duration was **1,105.927 s**, throughput **49.537 source examples/s**. Latest checkpoint and metadata were saved at **06:56:24 MDT**; metadata confirms 137 completed epochs.

At the **06:57 MDT** poll, Mamba was active in epoch **138**, step **12/214**, rolling loss **0.261156**, latest validation **0.298608**, LR **4.24700e-4**, EMA **0.998284**, ETA about **29h32m**, with RTX 5070 Ti use **6,173/16,303 MiB** (**14%**). ViT remains at **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 137. The figure index is `../runs/latent-wave/comparison/current/figure_index.md`. The validation loss at epoch 137 is above the selected best, so checkpoint selection remains based on epoch 136.


## Epoch 138 and refreshed figures — 2026-10-09 07:16 MDT

Mamba-2 seed 0 completed epoch **138/240** at global step **29,532**. Mean train JEPA loss was **0.258317123** and fixed-protocol validation loss was **0.296239384**. The selected best remains epoch **136** at **0.295680759**. Epoch duration was **1,116.192 s**, throughput **49.081 source examples/s**. Latest checkpoint and metadata were saved at **07:15:01 MDT**; metadata confirms 138 completed epochs and step 29,532.

At the **07:16 MDT** live poll, Mamba was active in epoch **139**, step **8/214**, rolling loss **0.261037**, latest validation **0.296239**, LR **4.18031e-4**, EMA **0.998301**, ETA about **29h16m**, with RTX 5070 Ti use **6,173/16,303 MiB** (**18%**). ViT remains **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 138. See `../runs/latent-wave/comparison/current/figure_index.md`. Epoch 138 validation improved versus epoch 137 but did not replace the epoch-136 best.


## Epoch 139 and refreshed figures — 2026-10-09 07:34 MDT

Mamba-2 seed 0 completed epoch **139/240** at global step **29,746**. Mean train JEPA loss was **0.254959116** and fixed-protocol validation loss was **0.295101906**, a new best (previous best: epoch 136, **0.295680759**). Epoch duration was **1,095.110 s**, throughput **50.026 source examples/s**. Latest, best checkpoint, and metadata were saved at **07:33:32 MDT**; metadata confirms 139 completed epochs and best validation loss **0.295101906**.

At the **07:34 MDT** poll, Mamba was active in epoch **140**, step **4/214**, rolling loss **0.249837**, latest validation **0.295102**, LR **4.11378e-4**, EMA **0.998317**, ETA about **29h00m**, with RTX 5070 Ti use **6,175/16,303 MiB** (**16%**). ViT remains at **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 139. See `../runs/latent-wave/comparison/current/figure_index.md`. The selected best checkpoint is now epoch 139.


## Epoch 140 and refreshed figures — 2026-10-09 07:53 MDT

Mamba-2 seed 0 completed epoch **140/240** at global step **29,960**. Mean train JEPA loss was **0.255490759** and fixed-protocol validation loss was **0.296518276**. The selected best remains epoch **139** at **0.295101906**. Epoch duration was **1,100.044 s**, throughput **49.802 source examples/s**. Latest checkpoint and metadata were saved at **07:52:11 MDT**; metadata confirms 140 completed epochs.

At the **07:53 MDT** poll, Mamba was active in epoch **141**, step **10/214**, rolling loss **0.252628**, latest validation **0.296518**, LR **4.04425e-4**, EMA **0.998334**, ETA about **28h43m**, with RTX 5070 Ti use **6,176/16,303 MiB** (**18%**). ViT remains at **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 140. See `../runs/latent-wave/comparison/current/figure_index.md`. Epoch 140 validation did not replace the epoch-139 best checkpoint.


## Epoch 141 and refreshed figures — 2026-10-09 08:12 MDT

Mamba-2 seed 0 completed epoch **141/240** at global step **30,174**. Mean train JEPA loss was **0.254881521** and fixed-protocol validation loss was **0.293175797**, a new best versus epoch 139 at **0.295101906** (about **0.65%** lower). Epoch duration was **1,101.658 s**, throughput **49.729 source examples/s**. Latest and best checkpoints plus metadata were saved at **08:10:40 MDT**; metadata confirms 141 completed epochs and the new best validation value.

At the **08:12 MDT** poll, Mamba was active in epoch **142**, step **16/214**, rolling loss **0.254676**, latest validation **0.293176**, LR **3.97491e-4**, EMA **0.998351**, ETA about **28h27m**, with RTX 5070 Ti use **6,175/16,303 MiB** (**15%**). ViT remains at **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 141. See `../runs/latent-wave/comparison/current/figure_index.md`. The selected best checkpoint is now epoch 141.


## Epoch 142 and refreshed figures — 2026-10-09 08:30 MDT

Mamba-2 seed 0 completed epoch **142/240** at global step **30,388**. Mean train JEPA loss was **0.253329852** and fixed-protocol validation loss was **0.293380436**. The selected best remains epoch **141** at **0.293175797**. Epoch duration was **1,106.970 s**, throughput **49.490 source examples/s**. The latest checkpoint and metadata were saved at **08:29:27 MDT**; metadata confirms 142 completed epochs and preserves the epoch-141 best validation loss.

At the **08:30 MDT** poll, Mamba was active in epoch **143**, step **2/214**, rolling loss **0.248850**, latest validation **0.293380**, LR **3.91206e-4**, EMA **0.998367**, ETA about **28h11m**, with RTX 5070 Ti use **6,177/16,303 MiB** (**17%**). ViT remains at **1/240**, queued behind Mamba; downstream results remain **0/8**.

Refreshed the training-history, validation, per-source, compute, downstream-status, and Table II PNGs and machine-readable summaries through completed epoch 142. See `../runs/latent-wave/comparison/current/figure_index.md`. Epoch 142 validation did not replace the epoch-141 best checkpoint.


## Current run update — 2026-10-09 08:52 MDT

Mamba-2 seed 0 remains active. Its latest completed epoch is **143/240** at global step **30,602**, with mean training JEPA loss **0.252679780**, fixed-protocol validation loss **0.293338094**, duration **1,107.976 s**, and throughput **49.445 source examples/s**. Best validation remains epoch **141** at **0.293175797**. At the latest live poll it was in epoch **144**, step **30,650/51,360** (**48/214**), rolling training loss **0.250131672**, LR **3.83062e-4**, EMA **0.998387**, elapsed **148,386 s**, ETA **100,263 s (~27h51m)**. The RTX 5070 Ti was at **6,178/16,303 MiB** and **20%** utilization; PyTorch allocated/reserved **2.26/4.32 GiB**.

The Mamba checkpoint is resumable and carries model identity, seed, resolved config, source revision, epoch/step, optimizer, scheduler, sampler, and RNG state. Run metadata confirms 143 completed epochs and the fixed validation protocol. The latest PNG snapshot is through completed epoch 143; it is partial. ViT seed 0 remains stopped after epoch 1 while its guarded handoff waits for Mamba; seed-0 downstream evaluation remains gated, with **0/8** result files. No local downstream metric or Mamba-versus-ViT performance claim is available yet. The four-source pretraining mix is unchanged, and `5g-csi.h5` remains excluded.

As a training-curve diagnostic, the fixed-protocol Mamba seed-0 aggregate JEPA train and validation losses are down **26.64%** and **16.20%**, respectively, from epoch 20 to epoch 143. Per-source validation loss is down for RFP (**15.10%**), WiFi CSI (**14.73%**), and NRPos Indoor (**30.84%**), but up for spectrogram (**22.21%**). The aggregate validation trend is improving, but this does not show a downstream win over the not-yet-completed ViT run; the spectrogram gap will remain a validation-only diagnostic after the matched primary comparison.


## Epoch 144 update — 2026-10-09 09:08 MDT

Mamba seed 0 completed epoch **144/240** (global step **30,816**) with mean train JEPA loss **0.249969702** and fixed-protocol validation loss **0.294683115**. Validation rose from 0.293338094 at epoch 143; the selected best remains epoch **141** at **0.293175797**. Epoch duration was **1,098.658 s** with throughput **49.864 source examples/s**. The active run is now in epoch **145**, step **30,830/51,360** (**14/214**), with rolling train loss **0.249739387**, LR **3.77444e-4**, EMA **0.998401**, and ETA about **27h37m**. The RTX 5070 Ti reported **6,179/16,303 MiB** used; PyTorch allocated/reserved **2.26/4.32 GiB**.

Figures were regenerated from completed logs through epoch 144. ViT remains at **1/240** and the eight seed-0 frozen-probe results remain **0/8**; the evaluation and later-seed queues are alive and waiting. Published paper scores are identified as references. The measured Mamba JEPA curve does not yet answer whether Mamba improves downstream results.


## Epoch 145 update — 2026-10-09 09:27 MDT

Mamba seed 0 completed epoch **145/240** at global step **31,030**. Mean train loss was **0.249879585** and fixed-protocol validation loss was **0.292463966**, a new best about **0.24%** below epoch 141 (**0.293175797**). Epoch duration was **1,110.799 s**, throughput **49.319 source examples/s**. Metadata and both checkpoints confirm epoch 145 and the fixed validation protocol. The active run is in epoch **146**, step **31,050/51,360** (**20/214**), with rolling train loss **0.249244526**, latest validation **0.292463966**, LR **3.70599e-4**, EMA **0.998418**, ETA about **27h20m**, and GPU use **6,184/16,303 MiB** (PyTorch allocated/reserved **2.26/4.32 GiB**).

The refreshed plots include completed Mamba epochs through 145. Across the fixed-protocol curve from epoch 20 to 145, aggregate train and validation JEPA losses decreased **27.46%** and **16.45%**. Validation fell for RFP, WiFi CSI, and NRPos Indoor, but rose **22.08%** for spectrogram. ViT remains at **1/240** and no downstream scores are available (**0/8**); therefore no Mamba-versus-ViT downstream conclusion is supported.


## Epoch 146 update — 2026-10-09 09:45 MDT

Mamba seed 0 completed epoch **146/240** at global step **31,244** with train loss **0.248082625** and fixed-protocol validation loss **0.287340588**, a new best **1.75%** below epoch 145. The checkpoint and run metadata confirm epoch 146, seed 0, Mamba-2, fixed validation, and resumable optimizer/scheduler state. Epoch duration was **1,097.211 s**, throughput **49.930 source examples/s**. The live run is now in epoch **147**, step **31,250/51,360** (**6/214**), rolling train loss **0.245899020**, latest validation **0.287340588**, ETA about **27h05m**. The RTX 5070 Ti reports **6,176/16,303 MiB**; PyTorch allocated/reserved **2.26/4.32 GiB**.

The refreshed figures cover completed logs through epoch 146. From fixed-protocol epoch 20 to 146, aggregate train and validation JEPA losses decreased **27.98%** and **17.91%**. Source validation decreased for RFP (**14.52%**), WiFi CSI (**17.39%**), and NRPos Indoor (**32.11%**), while it increased **20.16%** for spectrogram. ViT remains at **1/240** and the frozen-probe outputs remain **0/8**. No downstream model comparison is available yet.


## Live process check and ViT stop diagnosis — 2026-10-09 09:51 MDT

Mamba-2 seed 0 PID **1941478** remains active and unmodified in epoch **147/240**, step **31,320/51,360** (**76/214** steps this epoch). Rolling train loss is **0.245636397**; latest completed validation loss is **0.287340588** (epoch 146). LR is **3.62234e-4**, EMA is **0.998439**, ETA is approximately **27h00m**. PyTorch memory allocation/reservation is **2.26/4.32 GiB**; the RTX 5070 Ti reports **6,174/16,303 MiB** used at **17%** utilization.

The saved ViT seed-0 log records receipt of **SIGTERM**, followed by epoch-boundary checkpointing; metadata confirms a clean stop after epoch 1/240 at step 214. There is no traceback, and the log does not identify the signal sender. The sequential handoff remains alive and waits for Mamba's 240-epoch completion metadata before resuming the preserved ViT checkpoint with fixed validation. No downstream evaluation has run and no test split has been opened. The current PNG snapshot remains through completed Mamba epoch 146; epoch 147 is in progress and not counted as complete.

Six `nvidia-smi dmon` samples at **09:52 MDT** measured **14–18%** GPU SM use and **2–6%** memory-controller use while the Mamba trainer used about one CPU core (**104%**); each observed persistent DataLoader worker used less than **1%** CPU. The active config uses per-source batches of **8**, processes four source batches serially, and accumulates **8** such rounds, giving **32** source-microbatch forward/backward passes per optimizer update. This suggests Python/host launch overhead may contribute to low GPU occupancy; the run was not profiled, so this is an inference rather than a confirmed bottleneck. The configured primary run is unchanged. This throughput observation is separate from the unresolved spectrogram train/validation gap.

At **09:54 MDT**, Mamba was still active at epoch **147/240**, step **31,360** (**116/214**), train loss **0.246337250**, last completed validation **0.287340588**, LR **3.60998e-4**, EMA **0.998442**, ETA about **26h56m**, and GPU memory **6,174/16,303 MiB**. The guarded ViT handoff, seed-0 evaluation, matched seed-1/2, and fine-tuning queues were all live and gated; no extra GPU job was active, and seed-0 result files remained **0/8**. The fine-tuning queue log retains an earlier aborted invocation from when Mamba metadata reported only 98 epochs; its current invocation is waiting on the matched-seed queue. The plot snapshot remains through completed Mamba epoch 146.


## Epoch 147 results and refreshed figures — 2026-10-09 10:04 MDT

Mamba-2 seed 0 completed epoch **147/240** at global step **31,458** with train loss **0.246896229** and fixed-protocol validation loss **0.285505922**, a new best **0.64%** below epoch 146. The latest and best checkpoints were updated at **10:02 MDT**; metadata confirms the model, seed, completed epoch, step, and selected validation value. The active run is in epoch **148**, step **31,470/51,360** (**12/214**), rolling loss **0.247251548**, latest completed validation **0.285505922**, and ETA about **26h47m**. The GPU reports **6,893/16,303 MiB**; PyTorch allocation/reservation is **2.26/5.02 GiB**.

The refreshed PNGs and figure index now cover completed Mamba epochs through **147**. From fixed-protocol epoch 20 to 147, aggregate train and validation loss decreased **28.32%** and **18.43%**. Source validation changed **+22.75%** for spectrogram, **−18.52%** for RFP, **−18.31%** for WiFi CSI, and **−32.51%** for NRPos Indoor. The aggregate JEPA curve is improving, but the spectrogram source gap remains. ViT is still **1/240** and downstream outputs remain **0/8**; no Mamba-versus-ViT downstream comparison is available.

### Source-level validation diagnosis

The fixed-protocol spectrogram validation loss bottomed at **0.185053** on epoch **50** and rose to **0.234164** by epoch 147 while its training loss declined from **0.199881** at epoch 20 to **0.146406**. Its validation-minus-training gap changed from **−0.009117** to **+0.087758**. The source has **3,332** samples, so the fixed 5% validation split contains about **166** examples; the inventory found no repeated session/group IDs. The cache is already transformed and normalized, but the upstream fit scope of those statistics cannot be verified. This is consistent with spectrogram-specific overfitting or sample heterogeneity; available logs do not separate them. WiFi CSI's validation loss decreased overall, although its train/validation gap widened (**0.048195** to **0.071487**); RFP and NRPos Indoor show no comparable widening. There is no global representation-collapse signal in the available epoch-128 audit. These findings guide later validation-only tuning after the matched primary runs; they do not justify changing the active run or selecting with downstream test metrics.

At **10:06 MDT**, Mamba continued in epoch **148**, step **31,510/51,360** (**52/214**), with rolling train loss **0.245781113**, latest validation **0.285505922**, and ETA about **26h44m**. ViT remains at epoch 1; all evaluation queues remain gated, and the current PNG snapshot ends at completed Mamba epoch 147.


## Epoch 148 results and refreshed figures — 2026-10-09 10:22 MDT

Mamba seed 0 completed epoch **148/240** at step **31,672** with train loss **0.247056942** and fixed-protocol validation loss **0.285276087**, a new best **0.08%** below epoch 147. The process is active in epoch **149**, step **31,680/51,360** (**8/214**), rolling train loss **0.244748445**, validation **0.285276087**, ETA about **26h31m**. The RTX 5070 Ti reports **6,899/16,303 MiB** used; PyTorch allocated/reserved **2.26/5.02 GiB**.

Refreshed PNGs and the figure index cover Mamba epochs through **148**. From fixed-protocol epoch 20 to 148, aggregate train and validation loss decreased **28.28%** and **18.50%**. Per-source validation changes were spectrogram **+20.04%**, RFP **−19.47%**, WiFi CSI **−16.98%**, and NRPos Indoor **−32.75%**. Spectrogram validation improved from epoch 147 to 148, but remains above its epoch-50 low; WiFi CSI validation rose in epoch 148. These source-level issues remain the focus of later validation-only experiments after the matched baselines complete. ViT remains at **1/240**, and local downstream outputs remain **0/8**.

At **10:23 MDT**, the active Mamba process was at epoch **149**, step **31,690/51,360** (**18/214**), rolling train loss **0.248790540**, latest completed validation **0.285276087**, and ETA about **26h30m**. The plot snapshot remains through completed epoch **148**; the ViT handoff and all downstream queues are still gated.

At **10:25 MDT**, the Mamba-2 seed-0 process remained live at epoch **149**, step **31,720/51,360** (**24/214**), rolling mean train loss **0.245629126**, and latest completed fixed validation **0.285276087** from epoch 148. LR was **3.49919e-4**, EMA momentum **0.998470**, elapsed time **153,868 s**, and ETA **95,270 s (~26h27m)**. PyTorch allocated/reserved memory was **2.26/5.02 GiB**; the RTX 5070 Ti reported **6,899/16,303 MiB** and **16%** GPU utilization. No epoch boundary has occurred since the 10:21 figure refresh, so the plots still end at completed epoch 148. ViT remains stopped at epoch 1; its guarded resume queue is waiting for Mamba's 240-epoch completion. The downstream and later-seed queues remain gated; no local downstream scores or opened test splits are reported.

At **10:29 MDT**, Mamba was still active at epoch **149**, step **31,770/51,360** (**98/214**), with rolling training loss **0.244501449**, latest completed fixed validation **0.285276087** from epoch 148, LR **3.48387e-4**, EMA **0.998474**, elapsed time **154,117 s**, and ETA **95,032 s (~26h23m)**. GPU memory was **2.26/5.02 GiB** allocated/reserved in PyTorch and **6,899/16,303 MiB** on the RTX 5070 Ti (14% utilization). No local downstream results exist yet; the guarded ViT handoff and evaluation queues are waiting. Figures remain based on completed epoch 148.

At **10:32 MDT**, Mamba remained active at epoch **149**, step **31,800/51,360** (**128/214**), with rolling train loss **0.244726650** and latest completed fixed validation **0.285276087** (epoch 148). LR was **3.47468e-4**, EMA **0.998477**, elapsed time **154,265 s**, and ETA **94,888 s (~26h21m)**. PyTorch allocated/reserved memory was **2.26/5.02 GiB**; the RTX 5070 Ti reported **6,899/16,303 MiB** at **13%** utilization. Epoch 149 is incomplete; current figures end at epoch 148. The ViT resume and downstream queues remain gated, with no local downstream results.


## Epoch 149 results — 2026-10-09 10:41 MDT

Mamba-2 seed 0 completed epoch **149/240** at step **31,886** with mean training loss **0.244869640** and fixed-protocol validation loss **0.285385162**. This is **0.038%** higher than epoch 148 (**0.285276087**), so the selected checkpoint remains epoch **148**. Epoch duration was **1,086.875 s** and throughput **50.405 source examples/s**. The saved metadata confirms 149 completed epochs, fixed validation, Mamba-2, seed 0, and best epoch 148; the checkpoint stores epoch index 148 (zero-based) and global step 31,886.

All current training, source-loss, compute, downstream-status, and Table II comparison figures were refreshed from completed logs through epoch 149 at **10:40–10:42 MDT**. Summary/history/compute/Table II generators succeeded. The paper bars remain labeled reference values, and there are still **0/8** local downstream result files. The current `figure_index.md` is under `../runs/latent-wave/comparison/current/`.

Across fixed-validation epochs 20–149, aggregate train and validation loss decreased **28.91%** and **18.47%**. From epoch 148 to 149, validation increased slightly for spectrogram (**0.228986 → 0.231499**) and more noticeably for RFP (**0.106687 → 0.116372**), while it decreased for WiFi CSI (**0.427601 → 0.417990**) and NRPos Indoor (**0.377830 → 0.375679**). The RFP change is a validation-only signal to monitor, not a basis for selecting on downstream test results. No configuration was changed.

At **10:42 MDT**, Mamba had started epoch **150** and reached step **31,910/51,360** (**24/214**), with rolling train loss **0.246599556**, latest validation **0.285385162**, LR **3.44104e-4**, EMA **0.998485**, ETA **94,381 s (~26h13m)**, and RTX 5070 Ti memory **6,899/16,303 MiB**. ViT remains stopped after epoch 1 pending the guarded handoff; evaluation and later-seed queues remain gated. No downstream result or ranking is available.

At **10:45 MDT**, Mamba was active at epoch **150**, step **31,950/51,360** (**64/214**), with rolling train loss **0.244956455**, latest completed validation **0.285385162** (epoch 149), LR **3.42883e-4**, EMA **0.998488**, ETA **94,191 s (~26h10m)**, and RTX 5070 Ti memory **6,899/16,303 MiB**. The plots remain through completed epoch 149. No local downstream results are available; the ViT handoff and evaluation queues remain gated.

### Recent validation-window analysis — 2026-10-09 10:47 MDT

Compared fixed-protocol mean validation losses for epochs **130–139** and **140–149** to check whether the epoch-149 RFP increase is persistent. The aggregate and all four sources declined across these windows: aggregate **−2.38%**, spectrogram **−0.87%**, RFP **−1.35%**, WiFi CSI **−1.81%**, and NRPos Indoor **−4.18%**. In particular, RFP's epoch-149 point (**0.116372**) is above its prior epoch-148 best (**0.106687**), but its last-ten-epoch mean (**0.114543**) is lower than the preceding ten-epoch mean (**0.116115**). This is not evidence of a sustained RFP regression. Across-epoch standard deviations in the CSV are temporal variability, not example-level confidence intervals. The analysis uses fixed pretraining validation only; no downstream test data was used and no setting was changed. Values are in `../runs/latent-wave/comparison/current/validation_window_diagnostics.csv`.

At **10:47 MDT**, Mamba remained active at epoch **150**, step **31,970/51,360** (**84/214**), with rolling training loss **0.245479577**, completed validation **0.285385162**, ETA **94,096 s (~26h08m)**, and GPU memory **6,899/16,303 MiB**. The plots still end at completed epoch 149.

At **10:50 MDT**, Mamba seed 0 remained active at epoch **150**, step **32,000/51,360** (**114/214**), rolling training loss **0.245127737**, latest completed validation **0.285385162** (epoch 149), ETA **93,954 s (~26h06m)**, and RTX 5070 Ti memory **6,900/16,303 MiB**. No epoch has completed since the figure refresh, and downstream results remain unavailable.

At **10:51 MDT**, Mamba was active at epoch **150**, step **32,020/51,360** (**134/214**), rolling train loss **0.245113117**, latest validation **0.285385162** (epoch 149), ETA **93,857 s (~26h04m)**, and GPU memory **6,900/16,303 MiB**. The best checkpoint remains epoch 148, and no downstream scores are available.


## Epoch 150 results and refreshed figures — 2026-10-09 10:59 MDT

Mamba-2 seed 0 completed epoch **150/240** at step **32,100**. Mean training loss was **0.245051570**; fixed-protocol validation loss was **0.283024170**, a **0.79%** improvement over the former best epoch 148 (**0.285276087**). The selected validation checkpoint is now epoch **150**. Epoch duration was **1,098.940 s**, throughput **49.852 source examples/s**. Run metadata and checkpoint files confirm 150 completed epochs, seed 0, Mamba-2, fixed validation, and step 32,100.

Refreshed the current train/validation/source-loss, compute, downstream-status, and Table II reference figures and machine-readable summaries through epoch 150. The plotting and summary commands exited successfully. Local downstream result files remain **0/8**; the Table II bars are paper references, not results from these runs. Figure inventory: `../runs/latent-wave/comparison/current/figure_index.md`.

Across fixed-protocol epochs 20–150, aggregate training loss decreased **28.86%** and validation loss decreased **19.14%**. From epoch 149 to 150, spectrogram validation changed **0.231499 → 0.233668**, RFP **0.116372 → 0.107844**, WiFi CSI **0.417990 → 0.417367**, and NRPos Indoor **0.375679 → 0.373218**. The RFP rebound resolved at this epoch; this supports monitoring the broader fixed-validation trend rather than tuning from one validation point. No config change or downstream test inspection was made.

At **10:59 MDT**, Mamba was active in epoch **151**, step **32,110/51,360** (**10/214**), rolling train loss **0.245378260**, last completed validation **0.283024170**, ETA **93,447 s (~25h57m)**, and GPU use **6,903/16,303 MiB**. ViT remains at epoch 1 awaiting its guarded handoff; downstream and later-seed queues remain gated.

### Live status — 2026-10-09 11:08 MDT

Mamba-2 seed 0 remains in progress at epoch **151**, global step **32,210/51,360** (**110/214** steps in the epoch). Rolling training loss is **0.243771439**; the latest completed fixed-protocol validation loss is **0.283024170** from epoch **150**. LR is **3.34971e-4**, EMA momentum **0.998509**, elapsed time **156,368 s**, and ETA **92,966 s (~25h49m)**. PyTorch allocated/reserved memory is **2.26/5.02 GiB**; the RTX 5070 Ti reports **6,906/16,303 MiB** at **18%** utilization. Checkpoints and plotted curves remain through epoch 150.

The queue audit found the guarded ViT resume and all downstream/evaluation queues still waiting on the required 240-epoch completions. No local downstream result JSONs exist yet (**0/8** for seed 0). The evaluator is configured for the shared split manifest, target-encoder checkpoints, all four task identifiers, and split seeds 0–2; its queue uses validation-selected `best.pt` checkpoints. No test split has been opened.

## Epoch 151 results and refreshed figures — 2026-10-09 11:18 MDT

Mamba-2 seed 0 completed epoch **151/240** at global step **32,314**. Mean train loss was **0.243100424** and fixed-protocol validation loss was **0.281013859**, a **0.71%** improvement from epoch 150 (**0.283024170**); the best validation checkpoint is now epoch **151**. Epoch time was **1,091.211 s** at **50.205 source examples/s**. Checkpoint metadata confirms architecture Mamba-2, seed 0, the fixed validation protocol, 151 completed epochs, and best epoch 151.

Plots and machine-readable summaries refreshed at **11:17–11:20 MDT** include `pretraining_training_loss.png`, `pretraining_validation_loss.png`, `pretraining_validation_protocols.png`, `pretraining_source_losses.png`, `pretraining_compute_comparison.png`, `downstream_task_status.png`, and `paper_results_comparison.png`. The current run comparison remains partial: Mamba is **151/240**, ViT seed 0 is **1/240**, and there are **0/8** seed-0 downstream result files. Paper bars remain explicitly labeled reference results.

Aggregate fixed-protocol loss from epoch 20 to 151 decreased **29.42%** for training and **19.72%** for validation. The epoch-151 per-source validation losses were spectrogram **0.227466**, RFP **0.109710**, WiFi CSI **0.414881**, and NRPos Indoor **0.371998**. Versus epoch 150, three sources improved while RFP rose slightly. Adjacent ten-epoch validation means (132–141 vs. 142–151) decreased for the aggregate and all four sources; exact values and across-epoch deviations are in `../runs/latent-wave/comparison/current/validation_window_diagnostics.csv`. Spectrogram validation remains above its epoch-50 minimum. No downstream test data was used to select a checkpoint, and no config was changed.

At **11:20 MDT**, Mamba had advanced into epoch **152**, step **32,350/51,360** (**36/214**), with rolling training loss **0.241879780**, latest validation **0.281013859**, ETA about **25h38m**, and RTX 5070 Ti use **6,908/16,303 MiB** (14% utilization). The ViT resume, seed-0 probe, matched-seed, and fine-tuning jobs remain queued behind their completion gates.

## Epoch 152 results and refreshed figures — 2026-10-09 11:37 MDT

Mamba-2 seed 0 completed epoch **152/240** at global step **32,528**. Mean training loss was **0.242730679**; fixed-protocol validation loss was **0.280603312**, a **0.15%** improvement over epoch 151 (**0.281013859**). The best validation checkpoint is epoch **152**. Epoch duration was **1,093.273 s**, throughput **50.110 source examples/s**, and the run metadata confirms architecture Mamba-2, seed 0, fixed validation, 152 completed epochs, and best epoch 152.

The summary, training/validation/source-loss, validation-protocol, compute, downstream-status, and paper-comparison outputs were refreshed at **11:36–11:37 MDT**; all four commands exited successfully. Mamba is **152/240**, ViT seed 0 remains **1/240**, and downstream result JSONs remain **0/8**. The paper-score bars are reference results, not local measurements.

From epoch 20 to 152, aggregate training and fixed-validation loss decreased **29.53%** and **19.83%**. Epoch-152 source validation losses were spectrogram **0.228684**, RFP **0.107444**, WiFi CSI **0.416189**, and NRPos Indoor **0.370096**; compared with epoch 151, the first and third rose slightly while RFP and NRPos fell. The adjacent ten-epoch means for 133–142 versus 143–152 fell for the aggregate and all four sources. The CSV reports across-epoch variability, not confidence intervals. Spectrogram validation remains above its epoch-50 minimum. No training settings changed; downstream test data was not consulted.

At **11:37 MDT**, Mamba was active in epoch **153**, step **32,540/51,360** (**12/214**), with rolling train loss **0.243510827**, latest completed validation **0.280603312**, ETA about **25h23m**, and RTX 5070 Ti use **7,950/16,303 MiB** (18% utilization). ViT and downstream jobs remain gated behind completion checks.

### Live status — 2026-10-09 11:48 MDT

Mamba-2 seed 0 is active at epoch **153/240**, step **32,670/51,360** (**142/214** optimizer steps), with rolling training loss **0.241782635** and latest completed fixed-validation loss **0.280603312** from epoch **152**. LR is **3.21087e-4**, EMA momentum **0.998544**, elapsed time **158,716 s**, and live ETA **90,799 s (~25h13m)**. PyTorch allocated/reserved memory is **2.43/6.04 GiB**; physical RTX 5070 Ti use is **7,948/16,303 MiB** at **19%** utilization.

Checkpoint metadata and files remain at the last completed epoch: **152**, global step **32,528**, best epoch **152**. The guarded ViT resume, seed-0 evaluation, matched-seed, and fine-tuning queues are running and waiting; ViT remains **1/240**, with **0/8** seed-0 downstream JSONs. Since epoch 153 has no completed validation yet, current PNG curves correctly stop at epoch 152. No local downstream score or architecture comparison is available.

## Epoch 153 results and refreshed figures — 2026-10-09 11:55 MDT

Mamba-2 seed 0 completed epoch **153/240** at global step **32,742**, with mean train JEPA loss **0.241300251** and fixed-protocol validation loss **0.280087190**. This is **0.184%** below epoch 152 (**0.280603312**) and establishes a new best according to label-free pretraining validation. Epoch time was **1,095.250 s** with throughput **50.020 source examples/s**. Checkpoint metadata and files confirm epoch 153, seed 0, Mamba-2, global step 32,742, and fixed validation protocol.

The training summary, training and validation plots, per-source loss plots, protocol-separated validation plot, compute comparison, downstream-status PNG, Table II comparison, and figure index were refreshed at **11:55 MDT**; all four generators exited successfully. The validation-window diagnostics now compare epochs **134–143** with **144–153**. Local downstream outputs are still **0/8** and paper values remain clearly marked as references.

Aggregate fixed-protocol training loss fell **29.95%** and validation loss fell **19.98%** from epoch 20 to 153. Epoch-153 validation losses by source were spectrogram **0.227722**, RFP **0.110545**, WiFi CSI **0.413630**, and NRPos Indoor **0.368451**. Compared with epoch 152, RFP rose **2.89%** while the other three sources declined **0.42%**, **0.61%**, and **0.44%**, respectively. The ten-epoch means for aggregate and all four sources improved across the adjacent windows. This is validation-loss evidence and does not establish a downstream gain; no settings changed and no test data was consulted.

At **11:58 MDT**, Mamba was active in epoch **154**, step **32,790/51,360** (**48/214**), with rolling training loss **0.242361043**, latest completed validation **0.280087190**, LR **3.17490e-4**, EMA **0.998554**, elapsed time **159,346 s**, and ETA **90,243 s (~25h04m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,950/16,303 MiB** at **19%** utilization. The latest checkpoint remains epoch 153; ViT is **1/240**, and no downstream evaluation has run.

## Matched configuration audit — 2026-10-09 12:02 MDT

A read-only comparison of the prose and baseline configs and their resolved seed-0 configs confirmed that the shared protocol matches after excluding architecture-specific encoder settings and checkpoint output paths. Both configs select the same four HDF5 sources and keys, masking settings, per-source microbatch **8**, gradient accumulation **8**, effective aggregate batch **256**, **240** epochs, **214** optimizer steps per epoch, optimizer and LR/EMA schedules, weight decay, precision, workers, validation fraction, and seed **0**. The encoder factory dispatches by architecture; the Mamba-2 config subsection serialized in the resolved ViT YAML is not instantiated for the ViT branch.

The metadata records **19,085,440** total / **9,908,928** trainable parameters for Mamba-2 and **13,752,064** total / **7,242,240** trainable parameters for ViT, with the same **882,944**-parameter predictor. The data and optimizer protocol is matched, while encoder/backbone capacity differs; downstream results must be interpreted with that caveat. At the audit time, Mamba-2 was active at epoch **154**, step **32,830/51,360** (**88/214**). This audit did not modify the running job or its checkpoint.

## Epoch 154 results and refreshed figures — 2026-10-09 12:13 MDT

Mamba-2 seed 0 completed epoch **154/240** at global step **32,956**, with mean train JEPA loss **0.240666207** and fixed-protocol validation loss **0.280369333**. Validation was **0.101%** higher than epoch 153 (**0.280087190**), so the selected checkpoint remains epoch **153**; the latest checkpoint advanced to epoch 154. Metadata confirms the completed epoch, architecture, seed, and validation protocol. Epoch duration was **1,095.101 s**, throughput **50.026 source examples/s**.

The summary and training/validation/source-loss, validation-protocol, compute, downstream-status, and paper-comparison PNGs were regenerated at **12:13 MDT**. All four generators exited successfully. The validation-window CSV now compares fixed-validation epochs **135–144** with **145–154**. Local downstream outputs remain **0/8**, and paper results remain reference-only.

From epoch 20 to 154, aggregate fixed-protocol training loss decreased **30.13%** and validation loss decreased **19.90%**. From epoch 153 to 154, validation worsened slightly for spectrogram (**+0.10%**), RFP (**+1.08%**), and WiFi CSI (**+0.54%**), while improving for NRPos Indoor (**−0.68%**). The ten-epoch means declined for aggregate and each source across the adjacent windows. This is pretraining validation evidence only and does not demonstrate downstream improvement.

At **12:14 MDT**, Mamba was active in epoch **155**, step **32,970/51,360** (**14/214**), with rolling train loss **0.240850312**, latest validation **0.280369333**, LR **3.12116e-4**, EMA **0.998568**, elapsed **160,274 s**, and ETA **89,398 s (~24h50m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,948/16,303 MiB** at **17%** utilization. The best checkpoint remains epoch 153; ViT is **1/240** and downstream evaluation remains not run.

## PNG refresh — 2026-10-09 12:23 MDT

Regenerated the training summary, train/validation/source-loss and validation-protocol plots, compute plot, downstream task status, and paper-results comparison from the current logs. All four generators exited successfully. The PNGs contain completed epochs only: Mamba-2 seed 0 through epoch **154/240** and ViT seed 0 through epoch **1/240**. The comparison plot keeps published Table II scores labeled as paper references and leaves both local downstream score slots unevaluated. No downstream scores are available yet. The figure index is ../runs/latent-wave/comparison/current/figure_index.md.

At the live poll, Mamba-2 was still active at epoch **155**, step **33,070/51,360** (**114/214 optimizer steps**), with rolling mean training loss **0.239599820** and latest completed fixed-validation loss **0.280369333** from epoch 154. LR was **3.09142e-4**, EMA momentum **0.998576**, elapsed time **160,774 s**, and ETA **88,919 s (~24h42m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,946/16,303 MiB** at **16%** utilization. The latest checkpoint remains epoch 154 and the best-validation checkpoint remains epoch 153. The live process and checkpoints were left untouched.

## Epoch 155 results and refreshed figures — 2026-10-09 12:33 MDT

Mamba-2 seed 0 completed epoch **155/240** at global step **33,170**. Mean train JEPA loss was **0.239658090**; fixed-protocol validation loss was **0.281172887**, **0.287%** above epoch 154 (**0.280369333**). Best label-free validation remains epoch **153** at **0.280087190**. The latest checkpoint and metadata now confirm 155 completed epochs, seed 0, Mamba-2, global step 33,170, and validation protocol `fixed_seed_masks_channels_each_example_once_v1`. Epoch duration was **1,100.413 s** at **49.785 source examples/s**.

Regenerated the training summary, training/validation/source-loss and protocol plots, compute comparison, downstream status, and paper-results comparison; all four generators exited successfully. The plots show completed epochs only, through Mamba-2 epoch 155 and ViT epoch 1. Updated `validation_window_diagnostics.csv` to compare fixed-validation windows **136–145** and **146–155**. Aggregate window mean declined **4.05%**; source means declined for spectrogram (**2.07%**), RFP (**5.52%**), WiFi CSI (**3.82%**), and NRPos Indoor (**5.05%**). Across-epoch standard deviations are temporal variability, not confidence intervals.

From fixed-validation epoch 20 to 155, aggregate train loss fell **30.42%** (**0.344455 → 0.239658**) and validation loss fell **19.67%** (**0.350029 → 0.281173**). From epoch 154 to 155, validation increased for spectrogram (**0.227960 → 0.231894**) and WiFi CSI (**0.415844 → 0.418033**), and decreased for RFP (**0.111743 → 0.109507**) and NRPos Indoor (**0.365930 → 0.365257**). No setting changed, and no downstream test data was used.

At **12:33 MDT**, the trainer was active in epoch **156**, step **33,190/51,360** (**20/214 optimizer steps**), with rolling train loss **0.239541611**, last completed validation **0.281172887**, LR **3.05582e-4**, EMA **0.998585**, elapsed time **161,402 s**, and ETA **88,360 s (~24h33m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,945/16,303 MiB** at **18%** utilization. The latest checkpoint is epoch 155 and best remains epoch 153. ViT is **1/240** and downstream outputs are **0/8**; the gated sequential handoffs remain in place.

### Live resource and queue check — 2026-10-09 12:38 MDT

Mamba-2 seed 0 remains active at epoch **156**, step **33,240/51,360** (**70/214** optimizer steps), with rolling mean train loss **0.238868396** and most recently completed fixed-validation loss **0.281172887** from epoch 155. LR is **3.04103e-4**, EMA momentum **0.998589**, elapsed time **161,650 s**, and ETA **88,120 s (~24h28m)**. PyTorch allocated/reserved memory is **2.43/6.04 GiB**; the RTX 5070 Ti reports **7,945/16,303 MiB** at **17%** utilization. The C: volume has **638 GB free**. The guarded ViT continuation, seed-0 evaluation, matched seed-1/2, and fine-tuning queues remain alive and gated; local downstream results remain **0/8**. The trainer, checkpoints, and validation protocol were left unchanged.

## Epoch 156 results and refreshed figures — 2026-10-09 12:52 MDT

Mamba-2 seed 0 completed epoch **156/240** at global step **33,384**. Mean train JEPA loss was **0.239311758**; fixed-protocol validation loss was **0.283850867**, **0.952%** above epoch 155 (**0.281172887**). The best label-free validation checkpoint remains epoch **153** at **0.280087190**. Metadata and the updated `latest.pt` confirm 156 completed epochs, seed 0, architecture Mamba-2, and global step 33,384. Epoch duration was **1,088.092 s**, throughput **50.349 source examples/s**.

Refreshed the training summary, training/validation/source-loss and protocol PNGs, compute comparison, downstream status, and paper-results comparison; all four generators exited successfully. They plot completed epochs only, through Mamba-2 epoch 156 and ViT epoch 1. Updated `validation_window_diagnostics.csv` to compare fixed-validation windows **137–146** and **147–156**. The latter ten-epoch mean declined versus the prior window for aggregate validation (**−3.90%**), spectrogram (**−1.36%**), RFP (**−5.78%**), WiFi CSI (**−3.72%**), and NRPos Indoor (**−5.05%**). These are temporal window averages, not confidence intervals.

From fixed-protocol epoch 20 to 156, aggregate training loss fell **30.52%** (**0.344455 → 0.239312**) and validation loss fell **18.91%** (**0.350029 → 0.283851**). From epoch 155 to 156, per-source validation rose for spectrogram (**0.231894 → 0.238331**), RFP (**0.109507 → 0.109870**), and WiFi CSI (**0.418033 → 0.422937**), while NRPos Indoor declined (**0.365257 → 0.364265**). No settings changed and no downstream test data was used.

At **12:52 MDT**, the trainer was active in epoch **157**, global step **33,400/51,360** (**10/214 optimizer steps**), with rolling mean training loss **0.248569509**, latest completed validation **0.283850867** from epoch 156, LR **2.99382e-4**, EMA **0.998601**, elapsed **162,473 s**, and ETA **87,366 s (~24h16m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,943/16,303 MiB** at **17%** utilization. The latest checkpoint is epoch 156, while the best remains epoch 153. ViT remains **1/240**; local downstream outputs remain **0/8** and the sequential queues are gated.

## Read-only utilization sample — 2026-10-09 12:55 MDT

While Mamba-2 seed 0 was active at epoch **157**, step **33,430/51,360** (**46/214** optimizer steps), a six-second read-only sample observed GPU SM utilization of **15–21%** and memory-controller utilization of **2–7%**. The trainer process showed about **104% CPU** averaged over its lifetime; its 16 DataLoader workers each showed **0.2–0.6% CPU** and were mostly waiting in poll_schedule_timeout. This suggests low GPU occupancy and possible main-process CPU overhead, but the brief GPU sample does not isolate the bottleneck. HDF5 inputs remain on the /mnt/c mount. I did not profile by attaching instrumentation or change the active process/config/checkpoint; defer any loader or preprocessing optimization until the controlled run is safely complete.

At this poll, rolling train loss was **0.242828719**, latest completed fixed validation was **0.283850867** (epoch 156), LR **2.98499e-4**, EMA **0.998604**, ETA **87,221 s (~24h14m)**, and PyTorch allocated/reserved GPU memory **2.43/6.04 GiB**. Metadata and latest.pt remain at completed epoch **156**; best validation remains epoch **153**. The ViT and evaluation queues remain gated.

## Training-loop performance audit — 2026-10-09 13:01 MDT

A brief read-only profiling pass while the active Mamba-2 run continued found the trainer process averaging about **104% CPU** over its lifetime. Its **16** DataLoader workers each showed only **0.2–0.6% CPU** and were mostly waiting in `poll_schedule_timeout`. Six one-second GPU samples showed **15–21% SM** and **2–7% memory-controller** utilization. Source inspection of `src/pretrain_jepa.py` found that the inner source/group loop converts each detached loss to a Python `float` and performs a tensor finiteness check on every group before backward. Host scalar conversion and boolean evaluation of CUDA tensors can force device synchronization; together with 1,712 microbatches per epoch across four sources, this is a plausible source of Python/kernel-launch overhead and low occupancy. This observation is suggestive, not a confirmed bottleneck diagnosis. No instrumentation was attached, no code/config/checkpoint was changed, and the active training process was left untouched. If an optimization is pursued later, accumulate detached metrics on-device and reduce the frequency of synchronization checks, then validate it on a controlled smoke run and apply the same trainer code to both architectures.

At the same poll, Mamba-2 seed 0 was active at epoch **157**, global step **33,510/51,360** (**126/214 optimizer steps**), with rolling train loss **0.240740232**, latest fixed-validation loss **0.283850867** (epoch 156), LR **2.96149e-4**, EMA **0.998610**, elapsed time **163,008 s**, and ETA **86,831 s (~24h07m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,945/16,303 MiB** at **15%** utilization. The last completed checkpoint remains epoch **156**; best validation remains epoch **153**. No downstream task has run locally yet.

## Epoch 157 results and refreshed figures — 2026-10-09 13:10 MDT

Mamba-2 seed 0 completed epoch **157/240** at global step **33,598**. Mean train JEPA loss was **0.239127194**; fixed-protocol validation loss was **0.277606838**, down **2.20%** from epoch 156 (**0.283850867**). This is a new label-free best, improving the prior epoch-153 best (**0.280087190**) by **0.89%**. Metadata and both checkpoint timestamps confirm epoch 157 as latest and best, seed 0, architecture Mamba-2, and global step 33,598. Epoch duration was **1,087.083 s**, throughput **50.395 source examples/s**.

Refreshed the training summary and training/validation/source-loss/protocol, compute, downstream-status, and paper-comparison PNGs. All four generators exited successfully. Plots show completed records through Mamba-2 epoch 157 and ViT epoch 1. Updated validation-window diagnostics to fixed-validation windows **138–147** versus **148–157**. The latter ten-epoch mean declined for aggregate validation (**−3.74%**) and for spectrogram (**−1.63%**), RFP (**−4.71%**), WiFi CSI (**−3.48%**), and NRPos Indoor (**−4.99%**). These standard deviations describe temporal variation across epochs, not uncertainty across seeds.

From fixed-protocol epoch 20 to 157, aggregate train loss declined **30.58%** (**0.344455 → 0.239127**) and validation loss declined **20.69%** (**0.350029 → 0.277607**). From epoch 156 to 157, per-source validation improved for spectrogram (**0.238331 → 0.226680**), RFP (**0.109870 → 0.109679**), WiFi CSI (**0.422937 → 0.412976**), and NRPos Indoor (**0.364265 → 0.361092**). No training settings changed and no downstream test data was used.

At **13:14 MDT**, the trainer was active in epoch **158**, global step **33,660/51,360** (**62/214 optimizer steps**), with rolling train loss **0.236764485**, latest completed validation **0.277606838** from epoch 157, LR **2.91756e-4**, EMA **0.998621**, elapsed **163,784 s**, and ETA **86,125 s (~23h55m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,944/16,303 MiB** at **14%** utilization. ViT remains **1/240**, local downstream outputs remain **0/8**, and the GPU-sequential queues remain gated.


## Live-state reconciliation — 2026-10-09 13:14 MDT

The older `mamba2_prose_seed0/train_console.log` is a separate, stale run log ending at epoch 20; it is not the active trainer's output. PID **1941478** writes to `mamba2_prose_seed0_deterministic_val/train_console.log` under the fixed validation protocol. Its active record is epoch 158, step 33,660/51,360. `run_metadata.json` reports 157 completed epochs. `latest.pt` stores zero-based `epoch=156` (human epoch 157), global step 33,598, architecture Mamba-2, seed 0, best/last validation loss 0.277606838, and the fixed validation protocol; it also contains optimizer, scheduler, sampler, RNG, resolved-config, and Git-revision state for resuming. `best.pt` selects the same human epoch 157 using this label-free validation loss. On the fixed-protocol `train.jsonl`, epoch 20 to 157 training loss decreased **30.58%** (0.344455 to 0.239127) and validation loss decreased **20.69%** (0.350029 to 0.277607). This evidence does not indicate a stalled or non-improving run, so the active process remains uninterrupted. The guarded ViT handoff, seed-0 evaluation gate, matched-seed queue, and fine-tuning queue are present; downstream results are still **0/8**.


### Live poll — 2026-10-09 13:16 MDT

Mamba-2 seed 0 remained active at epoch **158**, global step **33,680/51,360** (**82/214** optimizer steps). Rolling train loss was **0.237325640**; latest completed fixed-validation loss was **0.277606838** from epoch 157. LR was **2.91172e-4**, EMA **0.998623**, elapsed **163,886 s**, and ETA **86,030 s (~23h54m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,944/16,303 MiB** at **19%** utilization. The latest and best checkpoints remain human epoch **157**; no active process or checkpoint was changed.


## Epoch 158 results and refreshed figures — 2026-10-09 13:29 MDT

Mamba-2 seed 0 completed epoch **158/240** at global step **33,812**. Mean training JEPA loss was **0.238264794**; fixed-protocol validation loss was **0.276105754**, down **0.541%** from epoch 157 (**0.277606838**), establishing a new label-free best. Metadata reports 158 completed epochs, best epoch 158, and global step 33,812. The checkpoint stores zero-based `epoch=157` (human epoch 158), architecture Mamba-2, seed 0, and validation protocol `fixed_seed_masks_channels_each_example_once_v1`; latest and best checkpoints include optimizer, schedulers, sampler/RNG states, resolved config, and Git revision. Epoch duration was **1,084.843 s**, throughput **50.499 source examples/s**.

Regenerated the pretraining summary, training/validation/protocol/source-loss curves, compute comparison, downstream status, and paper Table II comparison; all four generator commands exited successfully. Figures plot completed epochs through Mamba 158 and ViT 1. Paper values remain labeled as references; local downstream outputs remain **0/8**. Updated `validation_window_diagnostics.csv` to compare fixed-validation windows **139–148** and **149–158**. The latter window mean declined for aggregate validation (**−3.69%**), spectrogram (**−1.30%**), RFP (**−4.10%**), WiFi CSI (**−3.71%**), and NRPos Indoor (**−4.98%**). These across-epoch deviations are temporal variability, not confidence intervals.

From fixed-protocol epoch 20 to 158, aggregate training loss fell **30.83%** (**0.344455 → 0.238265**) and validation loss fell **21.12%** (**0.350029 → 0.276106**). Compared with epoch 157, validation rose slightly for spectrogram (**0.226680 → 0.227347**, **+0.29%**) and WiFi CSI (**0.412976 → 0.413312**, **+0.08%**), while RFP declined (**0.109679 → 0.104442**, **−4.77%**) and NRPos Indoor declined (**0.361092 → 0.359321**, **−0.49%**). No config changed and no downstream test data was used.

At **13:30 MDT**, the trainer was active in epoch **159**, global step **33,840/51,360** (**28/214** optimizer steps), with rolling mean training loss **0.238182796**, latest completed validation **0.276105754** from epoch 158, LR **2.86511e-4**, EMA **0.998636**, elapsed **164,706 s**, and ETA **85,273 s (~23h41m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,947/16,303 MiB** at **19%** utilization. The latest and best checkpoint is human epoch **158**. The active trainer and checkpoints were left untouched.


## Epoch 159 results and refreshed figures — 2026-10-09 13:47 MDT

Mamba-2 seed 0 completed epoch **159/240** at global step **34,026**. Mean training JEPA loss was **0.236554295**; fixed-protocol validation loss was **0.275311459**, down **0.288%** from epoch 158 (**0.276105754**), establishing a new label-free best. Run metadata reports 159 completed epochs, best epoch 159, and global step 34,026. The latest and best checkpoints store zero-based `epoch=158` (human epoch 159), Mamba-2, seed 0, the fixed validation protocol, and optimizer, scheduler, sampler/RNG, resolved-config, and Git-revision state. Epoch duration was **1,087.179 s**, throughput **50.391 source examples/s**.

Regenerated the pretraining summary, training/validation/protocol/source-loss plots, compute comparison, downstream status, and paper Table II comparison; all four generator commands exited successfully. Figures show completed epochs through Mamba 159 and ViT 1. The Table II values remain paper references, not local results. Local downstream outputs remain **0/8**. Updated `validation_window_diagnostics.csv` to compare fixed-validation windows **140–149** and **150–159**. The latter window mean declined for aggregate validation (**−3.71%**), spectrogram (**−1.50%**), RFP (**−5.26%**), WiFi CSI (**−3.40%**), and NRPos Indoor (**−4.94%**); the deviations are across-epoch temporal variability.

From fixed-protocol epoch 20 to 159, aggregate training loss fell **31.33%** (**0.344455 → 0.236554**) and validation loss fell **21.35%** (**0.350029 → 0.275311**). From epoch 158 to 159, validation improved for spectrogram (**0.227347 → 0.226214**, **−0.50%**), WiFi CSI (**0.413312 → 0.412266**, **−0.25%**), and NRPos Indoor (**0.359321 → 0.358313**, **−0.28%**); RFP increased slightly (**0.104442 → 0.104453**, **+0.01%**). No config changed and no downstream test data was used.

At **13:48 MDT**, the trainer was active in epoch **160**, global step **34,050/51,360** (**24/214 optimizer steps**), with rolling training loss **0.233024730**, latest completed validation **0.275311459** from epoch 159, LR **2.80428e-4**, EMA **0.998652**, elapsed **165,774 s**, and ETA **84,274 s (~23h24m)**. PyTorch allocated/reserved memory was **2.43/6.04 GiB**; the RTX 5070 Ti reported **7,948/16,303 MiB** at **17%** utilization. The latest and best checkpoint is human epoch **159**. The active process and checkpoints were left untouched.


### Live poll — 2026-10-09 13:57 MDT

Mamba-2 seed 0 is active at epoch **160**, global step **34,160/51,360** (**134/214 optimizer steps**). Rolling training loss is **0.236360081**; latest completed fixed-validation loss is **0.275311459** from epoch 159. LR is **2.77258e-4**, EMA **0.998660**, elapsed **166,323 s**, and ETA **83,746 s (~23h16m)**. PyTorch allocated/reserved memory is **2.43/6.04 GiB**; RTX 5070 Ti reports **7,948/16,303 MiB** at **14%** utilization. Latest and best checkpoint remains human epoch **159**. Downstream results remain **0/8**; the GPU sequential queues are gated behind Mamba completion.


## Mamba-2 seed 0 update — epoch 160 — 2026-10-09 14:08 MDT

The fixed-protocol Mamba-2 run completed epoch **160/240** at step **34,240/51,360**. Mean training JEPA loss was **0.235797782**, validation JEPA loss was **0.277012286**, epoch duration **1,097.483 s**, throughput **49.918 source examples/s**. The validation loss is **0.618% higher** than epoch 159, leaving the best-validation checkpoint at epoch **159** with **0.275311459**. The newest resumable checkpoint is human epoch 160; metadata reports 160 completed epochs and step 34,240. Both checkpoints retain the model identity, seed, protocol, resolved config, optimizer/scheduler and RNG/sampler state needed for selection/resume.

Across fixed-validation epochs 20–160, training loss declined **31.54%** (**0.344455 → 0.235798**) and validation loss declined **20.86%** (**0.350029 → 0.277012**). The latest fixed-protocol ten-epoch window (151–160) has mean validation loss **0.279313**, down **3.47%** from epochs 141–150 (**0.289357**); this window statistic describes epoch-to-epoch temporal behavior, not a confidence interval. Epoch 160’s aggregate validation increase over epoch 159 came mainly from RFP (+7.60%), while spectrogram (+0.25%) and WiFi CSI (+0.04%) were nearly flat and NRPos Indoor declined (−0.52%). The one-epoch increase does not overturn the longer declining trend, and the checkpoint remains selected strictly by validation loss.

Refreshed the machine-readable run summary and PNGs at `../runs/latent-wave/comparison/current/` through the completed Mamba epoch 160 (ViT remains one completed epoch). This includes pretraining train/validation curves, protocol-separated validation, source losses, compute, downstream task status, and paper Table II comparison. Published bars remain labeled paper-reported references. No downstream score has been measured locally (**0/8** JSON outputs); therefore no Mamba-versus-ViT downstream conclusion is available. `validation_window_diagnostics.csv` now compares epochs 141–150 and 151–160.

At the **14:07 MDT** live poll, Mamba-2 seed 0 was active at epoch **161**, step **34,260/51,360** (**20/214**), rolling train loss **0.234479274**, last completed validation **0.277012286** from epoch 160, LR **2.74385e-4**, EMA **0.998668**, elapsed **166,857 s**, and ETA **83,282 s (~23h08m)**. RTX 5070 Ti memory use was **7,946/16,303 MiB** with **14%** utilization. The live Mamba process was left uninterrupted. The guarded ViT, seed-0 evaluation, matched-seed, and fine-tuning queues remain gated behind completion of their prerequisites.


## Mamba-2 seed 0 update — epoch 161 — 2026-10-09 14:25 MDT

Mamba-2 seed 0 completed epoch **161/240** at step **34,454/51,360**. Mean train loss was **0.234305593**, fixed-validation loss **0.277931820**, epoch duration **1,085.994 s**, and source throughput **50.446 examples/s**. Validation rose **0.332%** from epoch 160; the best-validation checkpoint remains epoch **159** at **0.275311459**. The current latest checkpoint is human epoch 161 (checkpoint field `epoch=160`), with Mamba-2/seed-0 identity, resolved config, optimizer/scheduler, sampler/RNG, and code revision state verified.

Against fixed-protocol epoch 20, epoch 161 mean training loss is down **31.98%** (**0.344455 → 0.234306**) and validation loss is down **20.60%** (**0.350029 → 0.277932**). The epoch-161 source validation changes from epoch 160 were spectrogram **+2.43%**, RFP **+0.48%**, WiFi CSI **−0.24%**, and NRPos Indoor **−0.38%**. The ten-epoch fixed-validation mean for epochs 152–161 is **0.279005**, down **3.17%** from epochs 142–151 (**0.288141**); across-epoch deviations are temporal variation, not confidence intervals. The two latest epochs are above the best checkpoint, so selection remains validation-based and fixed at epoch 159.

Refreshed local PNGs through completed Mamba epoch 161 and ViT epoch 1 in `../runs/latent-wave/comparison/current/`; published Table II bars remain paper-reported references. There are still **0/8** downstream result JSONs, so no local downstream comparison or improvement claim is supported yet. The current live Mamba run remains active; the ViT, evaluation, seed-1/2, and fine-tuning queues are still waiting on their prerequisite gates.

At **14:25 MDT**, Mamba-2 seed 0 had entered epoch **162**, step **34,460/51,360** (**6/214**), rolling train loss **0.234512481**, last completed validation **0.277931820** at epoch 161, LR **2.68669e-4**, EMA **0.998684**, elapsed **167,871 s**, ETA **82,328 s (~22h52m)**, and RTX 5070 Ti memory **7,948/16,303 MiB** at **15%** utilization.


## Mamba-2 seed 0 update — epoch 162 — 2026-10-09 14:43 MDT

Epoch **162/240** completed at step **34,668/51,360** with mean training loss **0.234029948**, fixed-protocol validation loss **0.272443010**, duration **1,096.786 s**, and throughput **49.950 source examples/s**. Validation improved **1.975%** over epoch 161 and advances the best-validation checkpoint to epoch **162**. Both `latest.pt` and `best.pt` were loaded and checked for architecture, seed, epoch/step, validation protocol, optimizer/scheduler, sampler/RNG, resolved configuration, and code revision state.

The fixed-protocol training and validation losses are **32.06%** and **22.17%** lower, respectively, than at epoch 20. The ten-epoch validation mean for epochs 153–162 is **0.278189**, **3.02%** below epochs 143–152 (**0.286863**); per-source declines were spectrogram **0.97%**, RFP **2.50%**, WiFi CSI **2.75%**, and NRPos Indoor **4.74%**. These epoch-window statistics reflect temporal variability and are not confidence intervals. No downstream test data was used for checkpoint selection.

Regenerated PNGs and machine-readable run summaries through Mamba epoch 162; ViT has one completed epoch. The four downstream tasks remain unevaluated locally (**0/8** result JSONs). Paper Table II values remain explicitly labeled as paper-reported references; there is no local downstream comparison yet.

At **14:43 MDT**, Mamba-2 seed 0 was active in epoch **163**, step **34,680/51,360** (**12/214**), rolling train loss **0.230426753**, last completed validation **0.272443010** at epoch 162, LR **2.62425e-4**, EMA **0.998701**, elapsed **168,997 s**, ETA **81,282 s (~22h34m)**, and RTX 5070 Ti memory **8,131/16,303 MiB** at **17%** utilization. The training process and its checkpoint remain untouched.


## Partial-checkpoint downstream results — 2026-10-09

Following the user's request to prioritize downstream evaluation, Mamba-2 seed 0 was gracefully stopped after completing epoch **163/240**. Its best fixed-protocol validation checkpoint is epoch **163**, loss **0.270749662** (global step **34,882**); train loss at that epoch was **0.231739416**. A verified immutable copy is stored with the results. The training run remains partial and resumable.

Frozen evaluation completed on all four tasks, with the target encoder frozen, token-mean pooling, training-only standardization, and downstream split seeds 0/1/2. Scores (mean ± SD across downstream splits): RFS **70.89 ± 1.14%**, 5G NR positioning **2.429 ± 0.015 m**, DeepMIMO beam **49.35 ± 0.87%**, DeepMIMO LoS/NLoS **91.94 ± 0.47%**. See [08_partial_checkpoint_downstream_evaluation.md](08_partial_checkpoint_downstream_evaluation.md) for exact commands, protocol, paper-reference interpretation, checkpoint hash, continuation command, and limitations. Results, tables, logs, and PNGs are under `../runs/latent-wave/`; no full-run result paths were overwritten.


## Resumed controlled pretraining — 2026-10-09

Mamba-2 seed 0 resumed from the verified epoch-163 `latest.pt` under the unchanged 240-epoch prose configuration. At the first logged progress point it was at epoch **164/240**, step **34,890/51,360** (8/214 updates), train loss **0.232863**, latest validation **0.270750** at epoch 163, LR **2.56509e-4**, EMA **0.998717**, ETA about **22h18m**, and PyTorch allocated/reserved GPU memory **1.59/2.12 GiB**. It passed CUDA preflight on `cuda:0` (RTX 5070 Ti). Console output continues to append to the existing training log. The run remains incomplete; subsequent ViT, evaluation, matched-seed, and fine-tuning jobs are gated sequentially.

Mamba-2 seed 0 completed resumed epoch **164/240** at step **35,096/51,360**: train loss **0.230289102**, validation loss **0.271787534**, throughput **52.471 examples/s**. Validation increased from epoch 163, so `best.pt` remains the epoch-163 checkpoint at **0.270749662**; `latest.pt` advances to epoch 164 for resume. Mamba entered epoch 165 and training continues.

At **15:38 MDT**, Mamba-2 seed 0 was active at epoch **165/240**, step **35,120/51,360** (24/214). Rolling train loss was **0.230469**, most recent validation was **0.271788** at epoch 164, LR **2.50082e-4**, EMA **0.998735**, ETA about **22h**, PyTorch allocated/reserved memory **2.06/3.59 GiB**, and RTX 5070 Ti utilization **14%**. The live trainer advanced normally; the latest resumable checkpoint remains epoch 164 and the best-validation checkpoint epoch 163.

At **15:39 MDT**, the Mamba trainer had advanced to epoch **165**, step **35,130/51,360**. Rolling loss was **0.229368**, latest validation **0.271788** from epoch 164, ETA about **21h58m**, and RTX 5070 Ti memory use was **5,408 MiB** at **19%** utilization. No other GPU process was active; ViT and evaluation remain gated behind Mamba's 240-epoch completion.


## Checkpoint/evaluation audit — 2026-10-09 15:43 MDT

Current resumable `latest.pt`: completed human epoch **164/240**, global step **35,096**, validation loss **0.271787534**. Current validation-selected `best.pt`: human epoch **163**, validation loss **0.270749662**, SHA-256 `b89ddd3fbac9b397f4ab246c393f4eeb1af50f1421f2873d05ac8a1198d704ad`. The immutable checkpoint evaluated previously has the same hash. Thus the partial downstream results correspond to the current best, despite the user-supplied epoch-159 value having since advanced.

All four frozen downstream tasks and their three saved split seeds were already evaluated from this exact best checkpoint. Scores remain RFS **70.8897 ± 1.1376%**, NR positioning **2.429386 ± 0.015405 m**, DeepMIMO beam **49.3503 ± 0.8664%**, and DeepMIMO LoS/NLoS **91.9373 ± 0.4681%**. Variability is across downstream split seeds for one partial pretraining checkpoint, not across pretraining seeds. The exact protocol, result JSONs, console log, and Table II comparison figure are documented in `08_partial_checkpoint_downstream_evaluation.md` and `../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/`. No second held-out scoring pass was run.

At 15:43 MDT, seed 0 continued in epoch **165/240**, step **35,170/51,360**, rolling train loss **0.230638**, last completed validation loss **0.271788** (epoch 164), LR **2.48692e-4**, EMA **0.998739**, and ETA about **21h55m**. PyTorch allocated/reserved VRAM was **2.06/3.59 GiB**; the latest `nvidia-smi` sample showed **5,412/16,303 MiB** at **14%** utilization and only the Mamba trainer as a GPU compute process. The ViT, full-probe, matched-seed, and fine-tuning queues remain gated. The paper-comparison figure uses the evaluated best checkpoint (epoch 163); current latest training has completed epoch 164.

At **15:45 MDT**, Mamba-2 seed 0 was active in epoch **165/240**, step **35,190/51,360** (94/214), with rolling train loss **0.230942**, latest validation **0.271788** from epoch 164, LR **2.48136e-4**, EMA **0.998741**, and ETA **78,810 s (~21h53m)**. PyTorch allocated/reserved VRAM was **2.06/3.59 GiB**. The RTX 5070 Ti reported **5,411/16,303 MiB** at **17%** utilization; it showed only the Mamba process as a GPU compute job. The current best remains epoch 163; latest remains epoch 164.


## Refreshed current comparison artifacts — 2026-10-09 15:48 MDT

Regenerated the current external comparison outputs from the authoritative Mamba and ViT run directories plus the existing four Mamba epoch-163 frozen-probe result JSONs. The main PNGs are refreshed through **completed Mamba epoch 164** and ViT epoch 1; epoch 165 is in progress and appears only in this live status note. Current comparison output is `../runs/latent-wave/comparison/current/`; its `figure_index.md` inventories all eight PNGs and marks local Mamba scores as partial checkpoint results. The Table II figure now includes local Mamba values; Table III remains paper-reference-only. No additional downstream test scoring or GPU job was run.

The local Mamba score provenance is unchanged: the best checkpoint is epoch 163 with validation loss 0.270749662, and its SHA-256-verified copy was evaluated using the four persisted task split manifests and split seeds 0, 1, and 2. Those standard deviations describe split variation for one pretraining seed. ViT remains at 1/240, so there is no matched downstream comparison. The comparison report explicitly records this limitation and the latest Mamba epoch 164 state.

Commands used to refresh the CSV/JSON/PNG artifacts:

```bash
./.venv/bin/python -u src/summarize_training_runs.py \
  --run-dirs \
  ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
  ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val \
  --downstream-results \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_rfs.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_pos.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_deepmimo-beam.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_deepmimo-los.json \
  --out-dir ../runs/latent-wave/comparison/current

./.venv/bin/python -u src/plot_training_history.py \
  --run-dirs \
  ../runs/latent-wave/full_runs/mamba2_prose_seed0_deterministic_val \
  ../runs/latent-wave/full_runs/vit_prose_seed0_deterministic_val \
  --out-dir ../runs/latent-wave/comparison/current

./.venv/bin/python -u src/plot_compute_comparison.py \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --out-dir ../runs/latent-wave/comparison/current

./.venv/bin/python -u src/plot_paper_comparison.py \
  --paper-reference docs/mamba2_experiment/paper_results_reference.csv \
  --pretraining-runs ../runs/latent-wave/comparison/current/pretraining_runs.csv \
  --downstream-status ../runs/latent-wave/comparison/current/downstream_status.csv \
  --downstream-results \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_rfs.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_pos.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_deepmimo-beam.json \
  ../runs/latent-wave/results/partial_checkpoint_e163_fixed_validation/mamba2_pretrain_seed0_deepmimo-los.json \
  --out-dir ../runs/latent-wave/comparison/current
```

At the 15:48 MDT poll, Mamba was active at epoch **165/240**, step **35,230/51,360**, rolling train loss **0.230197**, validation loss **0.271788** (epoch 164), LR **2.47027e-4**, EMA **0.998744**, and ETA **78,616 s (~21h50m)**. PyTorch memory was **2.17/3.59 GiB allocated/reserved**; `nvidia-smi` showed **5,412/16,303 MiB** at **15%** utilization and only the Mamba process on the GPU.

The queued ViT resume was checked in its shell script and the trainer source. It waits for Mamba's 240-epoch completion and an idle GPU, verifies the epoch-1 checkpoint copy, then resumes with the supported `--reset-validation-selection` flag. That flag clears the legacy validation score used for prior selection while retaining training state. For seed 0 the two configs produce the same per-dataset unlabeled validation split and fixed validation masks/channels; the old ViT epoch-1 loss remains a separate legacy-protocol point, not a comparable score. The fixed-protocol ViT evaluations after resume can be compared with Mamba's fixed-protocol validation history.

## Read-only throughput diagnosis — 2026-10-09 15:52 MDT

A five-second GPU sample showed **16–20% SM** and **3–5% memory-controller** utilization at **74–75 W**. The Mamba Python trainer used **102% CPU** (about one core), while its HDF5 DataLoader workers were mostly at **0.2–0.6% CPU**. Source inspection found likely CPU-side synchronization in `src/pretrain_jepa.py`: every channel-count loss group checks CUDA finiteness and converts the loss to a Python float; then each parameter's gradient finiteness is separately converted to a host boolean before the optimizer step. `compute_loss_groups` may emit multiple groups for a batch. These are likely throughput costs, not a confirmed model-quality bug; HDF5 reads from the Windows-mounted `/mnt/c` path may contribute as well. The active process already has the module loaded, so I left its code untouched. Any synchronization optimization should be screened after the controlled baseline with the same step count and resolved configuration, then applied consistently to both architectures if it changes execution behavior.

At 15:52 MDT, Mamba seed 0 was active at epoch **165/240**, step **35,270/51,360**, rolling train loss **0.230194**, latest validation **0.271788** from epoch 164, LR **2.45919e-4**, EMA **0.998747**, and ETA **78,421 s (~21h47m)**. Allocated/reserved VRAM was **2.17/3.59 GiB**; the RTX 5070 Ti reported **5,411/16,303 MiB** at **15%**. The current fixed-validation best remains epoch 163 (0.270750); latest checkpoint remains completed epoch 164 (0.271788). No training or queued process was modified.

