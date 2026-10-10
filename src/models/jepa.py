import torch
import copy
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

from configs import JEPAConfig
from models import Predictor
from .encoder import build_encoder
from .utils import apply_masks, repeat_interleave_batch, get_effective_rank, sample_channel_subsets
from .validation import ValidationModule
from schedulers import WarmupCosineSchedule, CosineWDSchedule


@torch.no_grad()
def update_target_encoder(context_encoder: nn.Module, target_encoder: nn.Module, momentum: float) -> None:
    """EMA named state; floating tensors blend and integer buffers copy exactly."""
    if not 0.0 <= momentum <= 1.0:
        raise ValueError(f"EMA momentum must be in [0, 1], got {momentum}.")
    for kind, source, target in (
        ("parameter", dict(context_encoder.named_parameters()), dict(target_encoder.named_parameters())),
        ("buffer", dict(context_encoder.named_buffers()), dict(target_encoder.named_buffers())),
    ):
        if source.keys() != target.keys():
            raise RuntimeError(f"Context and target encoder {kind} names differ.")
        for name, source_tensor in source.items():
            target_tensor = target[name]
            if source_tensor.shape != target_tensor.shape or source_tensor.dtype != target_tensor.dtype:
                raise RuntimeError(f"Context and target {kind} mismatch at {name}.")
            if target_tensor.is_floating_point() or target_tensor.is_complex():
                target_tensor.mul_(momentum).add_(source_tensor.detach(), alpha=1.0 - momentum)
            else:
                target_tensor.copy_(source_tensor)


class JEPA(ValidationModule):
    def __init__(self, config: JEPAConfig, iterations_per_epoch: int, eval_fns: list[callable], metrics: list[str]):
        super().__init__(eval_fns, metrics)
        CONFIG = config
        self.context_encoder = build_encoder(CONFIG.encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.predictor = Predictor(
            num_patches=(CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2,
            encoder_embed_dim=CONFIG.encoder.embed_dim,
            output_dim=CONFIG.encoder.embed_dim,
            predictor_embed_dim=CONFIG.predictor.embed_dim,
            depth=CONFIG.predictor.depth,
            num_heads=CONFIG.predictor.num_heads,
        )
        self.loss = nn.MSELoss()
        self.iterations_per_epoch = iterations_per_epoch

        self.ema_start = CONFIG.optimization.ema[0]
        self.ema_end = CONFIG.optimization.ema[1]
        self.total_steps = iterations_per_epoch * CONFIG.optimization.epochs
        self.num_patches = (CONFIG.encoder.img_size[0] // CONFIG.encoder.patch_size) ** 2

        self._freeze_target_encoder()
        self._assert_ema_compatible()
        self.target_encoder.eval()
        self.config = CONFIG

    def _freeze_target_encoder(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _assert_ema_compatible(self):
        for name, parameter in self.context_encoder.named_parameters():
            target = dict(self.target_encoder.named_parameters()).get(name)
            if target is None or target.shape != parameter.shape:
                raise RuntimeError(f"Context/target parameter mismatch at {name}.")
        for name, buffer in self.context_encoder.named_buffers():
            target = dict(self.target_encoder.named_buffers()).get(name)
            if target is None or target.shape != buffer.shape:
                raise RuntimeError(f"Context/target buffer mismatch at {name}.")

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def update_target(self, momentum: float) -> None:
        update_target_encoder(self.context_encoder, self.target_encoder, momentum)

    def forward(self, x, masks_enc, masks_pred):
        C = x.size(1)

        context_emb = self.context_encoder(x, masks_enc)
        latent_mask_pred = self.predictor(context_emb, C, masks_enc, masks_pred)

        with torch.no_grad():
            target_out = self.target_encoder(x)
            latent_mask_target = F.layer_norm(target_out, (target_out.size(-1),))
            B, _, D = latent_mask_target.shape

            latent_mask_target = latent_mask_target.reshape(B, C, -1, D)
            latent_mask_target = apply_masks(latent_mask_target, masks_pred)
            latent_mask_target = repeat_interleave_batch(latent_mask_target, B=B, repeat=len(masks_enc))
            latent_mask_target = latent_mask_target.reshape(latent_mask_target.size(0), -1, D)

        return latent_mask_pred, latent_mask_target, target_out

    def training_step(self, batches, _):
        losses = []
        mask_percentages = []
        latent_reps = []
        for batch_name, batch in batches.items():
            loss, mask_percentage, rep = self.training_step_per_patch(batch_name, batch)
            losses.append(loss)
            mask_percentages.append(mask_percentage)
            latent_reps.append(rep)

        total_loss = torch.stack(losses).mean()
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, logger=True)

        self.log(f"mask_percentage", np.mean(mask_percentages), on_step=True, on_epoch=True, logger=True)
    
        latent_reps = torch.cat(latent_reps, dim=0)
        effective_rank = get_effective_rank(latent_reps)
        self.log(f"effective_rank", effective_rank, on_step=True, on_epoch=True, logger=True)

        with torch.no_grad():
            per_dim_std = latent_reps.std(dim=0)
            mean_std = per_dim_std.mean().item()
            self.log(f"mean_std", mean_std, on_step=True, on_epoch=True, logger=True)

        return total_loss
        
    def training_step_per_patch(self, name, batch):
        loss, mask_percentage, latent_rep, c_prime = self.compute_loss_per_patch(batch)
        self.log(f'train_loss_{name}', loss, on_step=True, on_epoch=True, logger=True)
        self.log(f"num_channels_{name}", c_prime, on_step=True, on_epoch=True, logger=True)
        return loss, mask_percentage, latent_rep

    def compute_loss_per_patch(self, batch):
        groups = self.compute_loss_groups(batch)
        batch_size = batch[0].shape[0]
        loss = sum(group["loss"] * group["sample_count"] / batch_size for group in groups)
        mask_percentage = sum(group["mask_percentage"] * group["sample_count"] / batch_size for group in groups)
        latent_rep = torch.cat([group["latent_rep"] for group in groups], dim=0)
        mean_channels = sum(group["channels"] * group["sample_count"] for group in groups) / batch_size
        return loss, mask_percentage, latent_rep, mean_channels

    def compute_loss_groups(self, batch):
        """Return one graph per sampled channel count so the trainer can backward sequentially."""
        x, masks_enc, masks_pred = batch
        batch_size, max_channels = x.shape[:2]
        channel_counts = torch.randint(1, max_channels + 1, (batch_size,), device=x.device)
        groups = []
        for channel_count in torch.unique(channel_counts, sorted=True).tolist():
            rows = torch.nonzero(channel_counts == channel_count, as_tuple=False).flatten()
            group_x = x.index_select(0, rows)
            if channel_count < max_channels:
                group_x = sample_channel_subsets(group_x, channel_count)
            group_enc = [mask.index_select(0, rows) for mask in masks_enc]
            group_pred = [mask.index_select(0, rows) for mask in masks_pred]
            prediction, target, target_out = self(group_x, group_enc, group_pred)

            num_masks = len(group_pred)
            group_size = len(rows)
            per_sample_loss = F.mse_loss(prediction, target, reduction="none")
            per_sample_loss = per_sample_loss.reshape(num_masks, group_size, -1, per_sample_loss.shape[-1])
            per_sample_loss = per_sample_loss.mean(dim=(0, 2, 3))
            mask_percentage = 1 - len(group_enc[0][0]) / self.num_patches
            groups.append({
                "loss": per_sample_loss.mean(),
                "sample_count": group_size,
                "channels": channel_count,
                "mask_percentage": mask_percentage,
                "latent_rep": target_out.mean(dim=1).detach(),
            })
        return groups
    
    def on_train_batch_start(self, batch, batch_idx):
        wd = self.optimizers().param_groups[0]['weight_decay']
        self.log('weight_decay', wd, on_step=True, on_epoch=False, logger=True) 
    
    def _get_validation_tokens_reps(self, x):
        return self.target_encoder(x)

    def _pool_validation_reps(self, token_latent_reps):
        return token_latent_reps.mean(dim=1)

    def configure_optimizers(self):
        CONFIG = self.config
        param_groups = [
            {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in self.context_encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in self.predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=CONFIG.optimization.start_lr, weight_decay=CONFIG.optimization.ref_weight_decay)
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(CONFIG.optimization.warmup * self.iterations_per_epoch),
            start_lr=CONFIG.optimization.start_lr,
            ref_lr=CONFIG.optimization.ref_lr,
            final_lr=CONFIG.optimization.final_lr,
            T_max=int(CONFIG.optimization.epochs * self.iterations_per_epoch)
        )
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=CONFIG.optimization.ref_weight_decay,
            final_wd=CONFIG.optimization.final_weight_decay,
            T_max=int(CONFIG.optimization.epochs * self.iterations_per_epoch)
        )                
        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
            }, 
            {
                'scheduler': wd_scheduler,
                'interval': 'step',
            }
        ]
