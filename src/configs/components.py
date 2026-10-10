import yaml

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Mamba2Config(BaseSettings):
    d_state: int = 64
    d_conv: int = 4
    expand: int = 1
    headdim: int = 64
    directionality: Literal["bidirectional", "forward"] = "bidirectional"


class EncoderConfig(BaseSettings):
    img_size: tuple[int, int]
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    architecture: Literal["vit", "mamba2"] = "vit"
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0
    mamba2: Mamba2Config = Field(default_factory=Mamba2Config)


class PredictorConfig(BaseSettings):
    embed_dim: int
    depth: int
    num_heads: int


class ProjectorConfig(BaseSettings):
    num_blocks: int
    hidden_dim: int


class WandBConfig(BaseSettings):
    project_name: str
    key: str | None = Field(default=None, validation_alias="WANDB_KEY")


class PrettrainDatasetConfig(BaseSettings):
    name: str
    path: str
    key: str


class DatasetConfig(BaseSettings):
    pretrain_datasets: list[PrettrainDatasetConfig]
    rfs_path: str | None = None
    pos_path: str | None = None
    beam_path: str | None = None
    batch_size: int
    num_workers: int
    pin_mem: bool
    validation_fraction: float = 0.05


class MultiBlockMaskConfig(BaseSettings):
    allow_overlap: bool
    aspect_ratio_type: Literal["span_freq", "span_time", "random"]
    aspect_ratio: list[float]
    enc_mask_scale: list[float]
    min_keep: int
    num_enc_masks: int
    num_pred_masks: int
    pred_mask_scale: list[float]


class RandomMaskConfig(BaseSettings):
    ratios: list[float]


class MasksConfig(BaseSettings):
    type: str
    multi_block: MultiBlockMaskConfig
    random: RandomMaskConfig


class TwoViewsTransformsConfig(BaseSettings):
  output_size: tuple[int, int]
  crop: bool
  crop_scale: tuple[float, float]
  n_time_masks: int
  time_mask_max_width: int
  n_freq_masks: int
  freq_mask_max_width: int
  add_noise: bool
  noise_std: float

  
class CheckpointConfig(BaseSettings):
    dir: str
    every_n_epochs: int
