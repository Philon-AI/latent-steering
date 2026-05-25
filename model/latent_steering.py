from pydantic import BaseModel, Field

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .encoders import VJEPAEncoder, VJEPAEncoderConfig, AttentivePooler, AttentivePoolerConfig
from .modules import DiT, DiTConfig, SelfAttentionTransformer, SelfAttentionTransformerConfig

def swish(x):
    return x * torch.sigmoid(x)


_PAD_TOKEN = 0
_VISION_TOKEN = 1
_STEERING_TOKEN = 2
_STATE_TOKEN = 3
_ACTION_TOKEN = 4


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()

        half_dim = self.embed_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float,
            device=timesteps.device) * (math.log(10000.0) / half_dim)

        freqs = timesteps.unsqueeze(-1) * exponent.exp()

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)

        return enc


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)

        self.timestep_encoder = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape ({B},), but got {tuple(timesteps.shape)}. "
                "Input must be 1D to be replicated across the sequence length T."
            )

        action_features = self.W1(actions)
        time_features = self.timestep_encoder(timesteps)
        time_features = time_features.to(dtype=action_features.dtype)

        x = torch.cat([action_features, time_features], dim=-1)
        x = self.W2(x)
        x = swish(x)
        x = self.W3(x)
        return x
    
class ActionDecoder(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LatentSteering_Config(BaseModel):
    add_pos_embed: bool = Field(default=False, description="Whether to add learned positional embeddings to the input token sequence.")
    add_pad_embed: bool = Field(default=False, description="Whether to insert a learned padding embedding between past and future vision tokens.")
    hidden_size: int = Field(default=1024, description="Input embedding dimension.")
    max_seq_len: int = Field(default=1024, description="Maximum sequence length supported by the positional embedding.")
    action_dim: int  = Field(default=None, description="Dimensionality of a single action vector.")
    action_horizon: int = Field(default=None, description="Number of future action steps predicted per forward pass.")
    noise_beta_alpha: float = Field(default=1.5, description="Alpha parameter of the Beta distribution used to sample flow-matching timesteps.")
    noise_beta_beta: float = Field(default=1.0, description="Beta parameter of the Beta distribution used to sample flow-matching timesteps.")
    noise_s: float = Field(default=0.999, description="Scaling factor applied to Beta-distributed samples to produce the flow-matching timestep t = (1 - sample) * s.")
    num_timestep_buckets: int = Field(default=1000, description="Number of discrete buckets used to embed continuous diffusion timesteps.")
    num_inference_timesteps: int = Field(default=16, description="Number of denoising steps run during inference.")

    vision_encoder_past_cfg: VJEPAEncoderConfig = Field(..., description="V-JEPA 2 ViT that encodes past frames into per-frame visual token embeddings.")
    vision_encoder_future_cfg: VJEPAEncoderConfig = Field(..., description="V-JEPA 2 ViT that encodes future frames into per-frame visual token embeddings.")
    attentive_pooler_past_cfg: AttentivePoolerConfig = Field(..., description="Attentive pooler that compresses past V-JEPA token sequences into a fixed, smaller set of vision tokens.")
    attentive_pooler_future_cfg: AttentivePoolerConfig = Field(..., description="Attentive pooler that compresses future V-JEPA token sequences into a fixed, smaller set of vision tokens.")
    multimodal_self_attention_cfg: SelfAttentionTransformerConfig = Field(..., description="Self-attention transformer over the joint multimodal (vision/language) token sequence.")
    diffusion_model_cfg: DiTConfig = Field(..., description="DiT denoiser that predicts action-trajectory velocity, cross-attending to the vision context.")

    tune_multi_projector : bool = Field(default=True, description="Tune multi projector if True.")
    tune_vision_encoder: bool = Field(default=True, description="Tune vision encoder if True.")
    tune_attentive_pooler: bool = Field(default=True, description="Tune attentive pooler if True.")
    tune_multimodal_self_attention: bool = Field(default=True, description="Tune multimodal self attention if True.")
    tune_diffusion_model: bool = Field(default=True, description="Tune diffusion model if True.")


class LatentSteering(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config["model"]
        self.config = LatentSteering_Config.model_validate(self.config)

        self.beta_dist = Beta(self.config.noise_beta_alpha, self.config.noise_beta_beta)

        self.action_encoder = ActionEncoder(
            action_dim=self.config.action_dim, hidden_size=self.config.hidden_size)

        self.action_decoder = ActionDecoder(
            hidden_size=self.config.hidden_size, action_dim=self.config.action_dim)

        if self.config.add_pos_embed:
            self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.vision_encoder_past = VJEPAEncoder(
            config=self.config.vision_encoder_past_cfg)
        self.vision_encoder_future = VJEPAEncoder(
            config=self.config.vision_encoder_future_cfg)

        self.attentive_pooler_past = AttentivePooler(
            config=self.config.attentive_pooler_past_cfg)
        self.attentive_pooler_future = AttentivePooler(
            config=self.config.attentive_pooler_future_cfg)

        if self.config.add_pad_embed:
            self.pad_embedding = nn.Parameter(torch.zeros(self.config.hidden_size))
            nn.init.normal_(self.pad_embedding, mean=0.0, std=0.02)

        self.multimodal_self_attention = SelfAttentionTransformer(
            config=self.config.multimodal_self_attention_cfg)

        self.diffusion_model = DiT(
            config=self.config.diffusion_model_cfg)

        self.set_trainable_parameters(
            tune_multi_projector=self.config.tune_multi_projector,
            tune_vision_encoder=self.config.tune_vision_encoder,
            tune_attentive_pooler=self.config.tune_attentive_pooler,
            tune_multimodal_self_attention=self.config.tune_multimodal_self_attention,
            tune_diffusion_model=self.config.tune_diffusion_model,
        )

    def set_trainable_parameters(
        self,
        tune_multi_projector: bool = True,
        tune_vision_encoder: bool = True,
        tune_attentive_pooler: bool = True,
        tune_multimodal_self_attention: bool = True,
        tune_diffusion_model: bool = True,
    ):
        self.tune_multi_projector = tune_multi_projector
        self.tune_vision_encoder = tune_vision_encoder
        self.tune_attentive_pooler = tune_attentive_pooler
        self.tune_multimodal_self_attention = tune_multimodal_self_attention
        self.tune_diffusion_model = tune_diffusion_model

        for param in self.parameters():
            param.requires_grad = True

        if not tune_multi_projector:
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.config.add_pad_embed:
                self.pad_embedding.requires_grad_(False)

        if not tune_vision_encoder:
            self.vision_encoder_past.requires_grad_(False)
            self.vision_encoder_future.requires_grad_(False)

        if not tune_attentive_pooler:
            self.attentive_pooler_past.requires_grad_(False)
            self.attentive_pooler_future.requires_grad_(False)

        if not tune_multimodal_self_attention:
            self.multimodal_self_attention.requires_grad_(False)

        if not tune_diffusion_model:
            self.diffusion_model.requires_grad_(False)

    def set_nontrainable_modules_eval(self):
        if self.training:
            if not self.tune_multi_projector:
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_vision_encoder:
                self.vision_encoder_past.eval()
                self.vision_encoder_future.eval()
            if not self.tune_attentive_pooler:
                self.attentive_pooler_past.eval()
                self.attentive_pooler_future.eval()
            if not self.tune_multimodal_self_attention:
                self.multimodal_self_attention.eval()
            if not self.tune_diffusion_model:
                self.diffusion_model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def encode_vision(self, past_images, future_images):
        if not self.tune_vision_encoder:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                past_features = self.vision_encoder_past(past_images)
                future_features = self.vision_encoder_future(future_images)
        else:
            past_features = self.vision_encoder_past(past_images)
            future_features = self.vision_encoder_future(future_images)
        past_features = self.attentive_pooler_past(past_features)
        future_features = self.attentive_pooler_future(future_features)
        return past_features, future_features

    def prepare_input_embs(self, mm_token_ids, sa_token_ids, past_vision_features, future_vision_features, action_features):
        B, T = mm_token_ids.shape
        mm_features = torch.cat([past_vision_features, future_vision_features], dim=1)
        mm_embs = torch.full(
            size=(B, T, self.config.hidden_size), fill_value=0.0, dtype=mm_features.dtype, device=mm_features.device
        )

        mm_mask = (mm_token_ids == _VISION_TOKEN) | (mm_token_ids == _STEERING_TOKEN)
        assert mm_mask.sum(dim=1).eq(mm_features.shape[1]).all(), (
            f"Each sample must have exactly {mm_features.shape[1]} combined "
            f"multimodal (_VISION_TOKEN and _STEERING_TOKEN) slots; "
            f"got counts {mm_mask.sum(dim=1).tolist()}"
        )

        batch_indices, token_indices = mm_mask.nonzero(as_tuple=True)
        mm_embs[batch_indices, token_indices] = mm_features.reshape(-1, self.config.attentive_pooler_past_cfg.embed_dim)

        if self.config.add_pad_embed:
            pad_mask = mm_token_ids == _PAD_TOKEN
            mm_embs[pad_mask] = self.pad_embedding.to(mm_embs.dtype)

        B, T = sa_token_ids.shape
        sa_embs = torch.full(
            size=(B, T, self.config.hidden_size), fill_value=0.0, dtype=action_features.dtype, device=action_features.device
        )

        action_mask = sa_token_ids == _ACTION_TOKEN
        action_mask = action_mask.unsqueeze(-1).expand_as(sa_embs)
        sa_embs = sa_embs.masked_scatter(action_mask, action_features)

        pos_ids = torch.arange(T, dtype=torch.long, device=sa_token_ids.device)
        if self.config.add_pos_embed:
            pos_embs = self.position_embedding(pos_ids)
            pos_embs = pos_embs.unsqueeze(0).expand(B, T, self.config.hidden_size)
            sa_embs = sa_embs + pos_embs

        return mm_embs, sa_embs

    def forward(self, data: dict) -> dict:
        self.set_nontrainable_modules_eval()

        past_vision_features, future_vision_features = self.encode_vision(
            data["past_images"], data["future_images"]
        )

        actions = data["future_actions"]

        t = self.sample_time(actions.shape[0],
            device=actions.device, dtype=actions.dtype
        )[:, None, None]
        t_discretized = (t[:, 0, 0] * self.config.num_timestep_buckets).long()

        noise = torch.randn_like(actions)
        noisy_trajectory = (1 - t) * noise + t * actions

        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        mm_embs, sa_embs = self.prepare_input_embs(
            data["mm_token_ids"],
            data["sa_token_ids"],
            past_vision_features,
            future_vision_features,
            action_features,
        )

        mm_embs = self.multimodal_self_attention(mm_embs)

        output, _ = self.diffusion_model(
            hidden_states=sa_embs,
            encoder_hidden_states=mm_embs,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )

        pred = self.action_decoder(output)
        pred_actions = pred[:, -actions.shape[1] :]

        velocity = actions - noise

        raw_loss = F.mse_loss(pred_actions, velocity, reduction="none")

        return {
            "raw_loss": raw_loss,
        }

    @torch.inference_mode()
    def get_action(self, data: dict) -> dict:
        batch_size = data["past_images"].shape[0]
        device = data["past_images"].device
        dtype = self.dtype

        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=dtype,
            device=device,
        )

        num_steps = self.config.num_inference_timesteps
        dt = 1.0 / num_steps

        past_vision_features, future_vision_features = self.encode_vision(
            data["past_images"], data["future_images"]
        )

        t_buckets = (torch.arange(num_steps, device=device,
            dtype=torch.float32) / num_steps * self.config.num_timestep_buckets).long()

        for i in range(num_steps):
            timesteps = t_buckets[i : i + 1].expand(batch_size)

            action_features = self.action_encoder(actions, timesteps)

            mm_embs, sa_embs = self.prepare_input_embs(
                data["mm_token_ids"],
                data["sa_token_ids"],
                past_vision_features,
                future_vision_features,
                action_features,
            )

            mm_embs = self.multimodal_self_attention(mm_embs)

            model_output = self.diffusion_model(
                hidden_states=sa_embs,
                encoder_hidden_states=mm_embs,
                timestep=timesteps,
            )

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -actions.shape[1] :]

            actions = actions + dt * pred_velocity

        return {
            "action_tensor": actions,
        }

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
