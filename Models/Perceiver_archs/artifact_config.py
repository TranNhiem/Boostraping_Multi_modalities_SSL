from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
from Models.Perceiver_archs.io_adapter import InputAdapter, OutputAdapter

@dataclass
class EncoderConfig:
    num_cross_attention_heads: int = 1
    num_cross_attention_qk_channels: Optional[int] = None   # num_latent_channels in PerceiverConfig (default)
    num_cross_attention_v_channels: Optional[int] = None    # num_latent_channels (default)
    num_cross_attention_layers: int = 1
    first_cross_attention_layer_shared: bool = False
    cross_attention_widening_factor: int = 1
    num_self_attention_heads: int = 8
    num_self_attention_qk_channels: Optional[int] = None   # num_latent_channels (default)
    num_self_attention_v_channels: Optional[int] = None    # num_latent_channels (default)
    num_self_attention_layers_per_block: int = 6
    num_self_attention_blocks: int = 8
    first_self_attention_block_shared: bool = True
    self_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02


@dataclass
class DecoderConfig:
    num_cross_attention_heads: int = 8
    num_cross_attention_qk_channels: Optional[int] = None
    num_cross_attention_v_channels: Optional[int] = None
    cross_attention_widening_factor: int = 1
    dropout: float = 0.0
    init_scale: float = 0.02


EnCfg = TypeVar("EnCfg", bound=EncoderConfig)
DeCfg = TypeVar("DeCfg", bound=DecoderConfig)

@dataclass
class PerceiverConfig(Generic[EnCfg, DeCfg]):
    enc_cfg: EnCfg
    dec_cfg: DeCfg
    num_latents: int = 512
    num_latent_channels: int = 1024
    activation_checkpointing: bool = False
    activation_offloading: bool = False
