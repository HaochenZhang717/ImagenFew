from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from simple_vae import SimpleVAE


def _maybe_cast_dtype(dtype_name: Optional[str]):
    if dtype_name is None or dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


class LatentSoftPromptProjector(nn.Module):
    def __init__(self, latent_dim: int, llm_hidden_size: int, soft_prompt_tokens: int) -> None:
        super().__init__()
        self.soft_prompt_tokens = soft_prompt_tokens
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, llm_hidden_size),
            nn.SiLU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.positional = nn.Parameter(torch.zeros(1, soft_prompt_tokens, llm_hidden_size))

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(latent_seq)
        if x.size(1) != self.soft_prompt_tokens:
            x = F.interpolate(
                x.transpose(1, 2),
                size=self.soft_prompt_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return x + self.positional


class Stage1LatentCaptionModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        vae_cfg = cfg["vae"]
        model_cfg = cfg["model"]
        training_cfg = cfg["training"]

        self.caption_use_mu = bool(model_cfg.get("caption_use_mu", True))

        dynamic_size = vae_cfg.get("dynamic_size", max(vae_cfg["input_dim"], vae_cfg["output_dim"]))
        self.vae = SimpleVAE(
            input_dim=vae_cfg["input_dim"],
            output_dim=vae_cfg["output_dim"],
            hidden_size=vae_cfg.get("hidden_size", 128),
            num_layers=vae_cfg.get("num_layers", 4),
            num_heads=vae_cfg.get("num_heads", 4),
            latent_dim=vae_cfg.get("latent_dim", 64),
            beta=vae_cfg.get("beta", 0.001),
            dynamic_size=dynamic_size,
            encoder_channels=vae_cfg.get("encoder_channels"),
            encoder_downsample_stages=vae_cfg.get("encoder_downsample_stages", 2),
            decoder_channels=vae_cfg.get("decoder_channels"),
            decoder_res_blocks=vae_cfg.get("decoder_res_blocks", 1),
            decoder_dropout=vae_cfg.get("decoder_dropout", 0.0),
            decoder_upsample_stages=vae_cfg.get("decoder_upsample_stages", 2),
            seq_len=vae_cfg["seq_len"],
        )

        torch_dtype = _maybe_cast_dtype(model_cfg.get("torch_dtype"))
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_cfg["llm_name"],
            torch_dtype=torch_dtype,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.llm.config.use_cache = False

        if model_cfg.get("gradient_checkpointing", False):
            self.llm.gradient_checkpointing_enable()
            self.llm.enable_input_require_grads()

        if model_cfg.get("use_lora", True):
            lora_config = LoraConfig(
                r=model_cfg.get("lora_r", 16),
                lora_alpha=model_cfg.get("lora_alpha", 32),
                lora_dropout=model_cfg.get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=model_cfg.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            )
            self.llm = get_peft_model(self.llm, lora_config)
        elif model_cfg.get("freeze_base_llm", False):
            for param in self.llm.parameters():
                param.requires_grad = False

        llm_hidden_size = self.llm.get_input_embeddings().embedding_dim
        self.soft_prompt = LatentSoftPromptProjector(
            latent_dim=vae_cfg.get("latent_dim", 64),
            llm_hidden_size=llm_hidden_size,
            soft_prompt_tokens=model_cfg.get("soft_prompt_tokens", 16),
        )
        self._base_requires_grad = {
            name: param.requires_grad for name, param in self.named_parameters()
        }

    @property
    def soft_prompt_tokens(self) -> int:
        return self.soft_prompt.soft_prompt_tokens

    def load_vae_weights(self, ckpt_path: str, map_location: str = "cpu") -> None:
        state = torch.load(ckpt_path, map_location=map_location)
        weights = state.get("model", state)
        self.vae.load_state_dict(weights, strict=False)

    def configure_trainable_modules(
        self,
        train_vae: bool,
        train_soft_prompt: bool,
        train_llm: bool,
    ) -> None:
        for name, param in self.named_parameters():
            base_flag = self._base_requires_grad[name]
            if name.startswith("vae."):
                param.requires_grad = base_flag and train_vae
            elif name.startswith("soft_prompt."):
                param.requires_grad = base_flag and train_soft_prompt
            elif name.startswith("llm."):
                param.requires_grad = base_flag and train_llm

    def _build_inputs_embeds(
        self,
        ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        vae_outputs = self.vae(ts)
        caption_latent = vae_outputs["mu"] if self.caption_use_mu else vae_outputs["z"]
        soft_prompt_embeds = self.soft_prompt(caption_latent)

        token_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([soft_prompt_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            attention_mask.size(0),
            self.soft_prompt_tokens,
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), self.soft_prompt_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        return vae_outputs, inputs_embeds, full_attention_mask, full_labels

    def forward(
        self,
        ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.joint_caption_step(
            ts=ts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            caption_loss_weight=1.0,
            kl_loss_weight=1.0,
        )

    def vae_pretrain_step(self, ts: torch.Tensor) -> Dict[str, torch.Tensor]:
        vae_outputs = self.vae(ts)
        vae_losses = self.vae.loss_function(
            ts,
            vae_outputs["recon"],
            vae_outputs["mu"],
            vae_outputs["logvar"],
        )
        return {
            "loss": vae_losses["loss"],
            "caption_loss": torch.zeros_like(vae_losses["loss"]),
            "vae_loss": vae_losses["loss"],
            "recon_loss": vae_losses["recon_loss"],
            "kl_loss": vae_losses["kl_loss"],
        }

    def joint_caption_step(
        self,
        ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        caption_loss_weight: float = 1.0,
        kl_loss_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        vae_outputs, inputs_embeds, full_attention_mask, full_labels = self._build_inputs_embeds(
            ts=ts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            use_cache=False,
        )
        vae_losses = self.vae.loss_function(
            ts,
            vae_outputs["recon"],
            vae_outputs["mu"],
            vae_outputs["logvar"],
        )
        kl_term = kl_loss_weight * vae_losses["kl_loss"]
        total_loss = caption_loss_weight * llm_outputs.loss + kl_term
        return {
            "loss": total_loss,
            "caption_loss": llm_outputs.loss,
            "vae_loss": kl_term,
            "recon_loss": vae_losses["recon_loss"],
            "kl_loss": vae_losses["kl_loss"],
        }

    @torch.no_grad()
    def generate(
        self,
        ts: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        generation_kwargs = generation_kwargs or {}
        _, inputs_embeds, full_attention_mask, _ = self._build_inputs_embeds(
            ts=ts,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            **generation_kwargs,
        )
