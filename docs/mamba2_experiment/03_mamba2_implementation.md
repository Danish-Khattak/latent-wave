# Mamba-2 implementation

Implementation status: CUDA tests and bounded smoke passed on 2026-10-05. No full pretraining was run.

## Model path

`encoder.architecture` selects either the original ViT encoder or `Mamba2Encoder` through `src/models/encoder.py`. Both implement the same input/output contract: `(B,C,224,224)` in and `(B,C×196,256)` tokens out. The existing shared one-channel 16×16 convolution patch projection is reused; a 224×224 channel produces 14×14 = 196 patches. Fixed 2-D sine/cosine positions are added to every channel's patch grid.

Tokens use the repository's existing deterministic order: channel-major, then row-major patch index. A spatial mask is applied to each channel before flattening. The target sees the full input, and the predictor receives the same target patch indices and channel ordering as before. There is no learned antenna embedding in the original ViT path, so none was added to the controlled comparison.

Each `Mamba2Block` has independent official forward and reverse `Mamba2` modules. The reverse path reads `tokens.flip(1)` and flips its output back before concatenation; a learned projection maps the concatenated paths to the encoder width. Pre-norm residuals and the existing MLP-style feed-forward branch remain. Bidirectional processing is the default because flattening a two-dimensional patch grid does not establish a causal sequence. `directionality: forward` is available for a documented ablation.

Initial parameters are explicit in the configs: width 256, depth 8, state size 64, convolution width 4, expansion 1, head dimension 64, bidirectional. This preserves the ViT width/depth and avoids assuming that the parameter count matches. The actual prose-default run metadata records:

| Count | Mamba-2 | ViT baseline |
|---|---:|---:|
| Context encoder | 9,101,248 | 6,434,560 |
| EMA target encoder | 9,101,248 | 6,434,560 |
| Predictor | 882,944 | 882,944 |
| Full JEPA module, including both encoders | 19,085,440 | 13,752,064 |
| Trainable parameters | 9,908,928 | 7,242,240 |

These counts come from the actual resolved prose configs and include both context and target encoders. The bidirectional Mamba-2 model has 38.8% more full-module parameters than the matched ViT config, while width, depth, data, masks, predictor, and effective batch are held fixed. The first result is therefore architecture-controlled but not parameter-count matched; compare downstream results per task and report this capacity difference.

## Variable channels, lengths, and state

JEPA samples each item a channel count uniformly from 1 through its available channel count. Samples with the same selected count are grouped, and the selected channel indices are sorted so antenna order remains stable. Each sequence is processed independently; no recurrent state is passed between samples or calls. The data collator makes equal-sized masks within a batch. The block also accepts per-row valid lengths and groups equal lengths before calling Mamba, then zeroes padded outputs; the CUDA test changes padding values and confirms valid outputs do not change.

Context and target are initialized as a deep copy and checked for matching named parameter/buffer shapes. Target gradients are disabled and its mode is forced to `eval()` even when the JEPA module enters training mode. EMA updates match names rather than positional `zip`: floating and complex state is blended, while integer buffers are copied. The trainer calls EMA only after a finite-gradient optimizer update. Tests cover target initialization, names/shapes, frozen state, EMA arithmetic, optimizer-before-EMA order, mask-to-target indexing, and checkpoint round-trip.

The predictor, target LayerNorm, target-mask indexing, and MSE objective remain in the existing JEPA module. Model-side changes are in `src/models/mamba2_encoder.py`, `src/models/encoder.py`, and `src/models/jepa.py`; the label-free update/checkpoint loop is in `src/pretrain_jepa.py`.

## Verified checks

- Official `Mamba2(d_model=256, d_state=64, d_conv=4, expand=1, headdim=64)` CUDA forward and backward both returned finite tensors and gradients.
- The HDF5-backed batch-8 smoke used the real NRPos Indoor `normalized_csi` loader, masks, Mamba context and target encoders, predictor, JEPA loss, student/predictor gradients, optimizer step, EMA, and checkpoint save.
- Peak PyTorch CUDA allocated memory for that bounded smoke was 1,407,824,896 bytes (about 1.31 GiB). This is a one-source, one-update measurement, not a full epoch or full-run guarantee.
- Batch-8, four-source-config preflights peaked at 361,396,224 allocated bytes for Mamba-2 and 98,454,016 bytes for ViT. These are inference preflight peaks, not training-memory comparisons.
- Both full four-source prose-default dry runs passed their CUDA model preflight with matching `(48, 28, 256)` predictor output.
- `uv run pytest -q`: 12 passed. See `06_implementation_log.md` for the scope and warning notes.
