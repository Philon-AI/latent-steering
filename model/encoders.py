import logging

from pydantic import BaseModel, Field

import torch
from torch import nn
from torchvision.transforms.v2 import functional as TF

import vjepa2.models_v2_1.vision_transformer as _VJEPAVIT
from vjepa2.models.attentive_pooler import AttentivePooler as _VJEPAAttentivePooler

logger = logging.getLogger(__name__)


class AttentivePoolerConfig(BaseModel):
    num_queries: int = Field(..., description="Number of pooled queries the AttentivePooler reduces the token count to.")
    embed_dim: int = Field(..., description="Embedding dimension of the vision encoder output.")
    num_heads: int = Field(..., description="Number of attention heads in the AttentivePooler.")
    depth: int = Field(..., description="Number of blocks in the AttentivePooler.")


class AttentivePooler(nn.Module):
    def __init__(self, config: AttentivePoolerConfig):
        super().__init__()
        self.config = config
        self.attentive_pooler = _VJEPAAttentivePooler(
            num_queries=config.num_queries,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            depth=config.depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attentive_pooler(x)


class VJEPAEncoderConfig(BaseModel):
    model_name: str = Field(..., description="V-JEPA 2 ViT variant to instantiate from vjepa2.models_v2_1.vision_transformer (e.g. 'vit_large').")
    pretrained_path: str | None = Field(default=None, description="Path to a V-JEPA 2 checkpoint .pt; weights under 'ema_encoder' are loaded into the encoder.")
    img_size: tuple[int, int] = Field(..., description="Input frame spatial size (H, W) after preprocessing, e.g. (384, 384) for the ViT-L/384 config.")
    img_temporal_dim_size: int = Field(default=1, description="Temporal size of single-image inputs treated as 1-frame clips; enables the V-JEPA 2.1 image-aware patch embedder.")
    num_frames: int = Field(..., description="Number of frames per video clip the encoder is configured for (e.g. 32 for the EK100 frozen eval).")
    tubelet_size: int = Field(default=2, description="Number of consecutive frames grouped into one spatiotemporal patch along the time axis.")
    patch_size: int = Field(default=16, description="Spatial patch size in pixels for tokenization; must divide both img_size dimensions.")
    uniform_power: bool = Field(default=True, description="If True, use uniform power scaling across spatial and temporal axes when constructing positional embeddings.")
    use_rope: bool = Field(default=True, description="If True, use rotary position embeddings (RoPE) instead of learned/sinusoidal positional embeddings.")


class VJEPAEncoder(nn.Module):
    def __init__(self, config: VJEPAEncoderConfig):
        super().__init__()

        self.config = config

        vit_kwargs = self.config.model_dump(exclude={"model_name", "pretrained_path"})
        self.vision_encoder = getattr(_VJEPAVIT, self.config.model_name)(**vit_kwargs)

        if self.config.pretrained_path:
            self._load_checkpoint(self.config.pretrained_path)

    def _load_checkpoint(self, path: str) -> None:
        logger.info("Loading V-JEPA encoder weights from %s", path)
        state = torch.load(path, map_location="cpu", weights_only=True)["ema_encoder"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = self.vision_encoder.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys: %d (first 3: %s)", len(missing), missing[:3])
        if unexpected:
            logger.warning("Unexpected keys: %d (first 3: %s)", len(unexpected), unexpected[:3])

    def _preprocess(
        self,
        images: torch.Tensor,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> torch.Tensor:
        B, N, C, H, W = images.shape
        target_h, target_w = self.config.img_size
        x = images.reshape(B * N, C, H, W).float()
        x = TF.resize(x, size=min(target_h, target_w), antialias=True)
        x = TF.center_crop(x, [target_h, target_w])
        x = TF.normalize(x / 255.0, mean=list(mean), std=list(std))
        return x.reshape(B, N, C, target_h, target_w)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self._preprocess(images)
        x = images.permute(0, 2, 1, 3, 4).contiguous()
        image_features = self.vision_encoder(x)
        return image_features
