import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
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
        if soft_prompt_embeds.dtype != token_embeds.dtype:
            soft_prompt_embeds = soft_prompt_embeds.to(dtype=token_embeds.dtype)
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


class Stage1LatentCaptionModelVE(Stage1LatentCaptionModel):
    SENTENCE_PATTERN = re.compile(r"[^.]+\.")
    NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
    LOCAL_SEGMENT_PATTERN = re.compile(r"^At the (beginning|middle|end):\s*(.+)\.$", re.IGNORECASE)
    PEAK_PREFIX_PATTERN = re.compile(r"^there are\s+\d+\s+peaks,\s*", re.IGNORECASE)

    def __init__(self, cfg: Dict, tokenizer=None) -> None:
        self.cfg = cfg
        self._ve_grad_hooks = []
        self._template_to_token: Dict[str, str] = {}
        self.vocab_expansion_summary: Dict = {
            "enabled": False,
            "num_added_tokens": 0,
            "token_to_template": [],
        }
        self._dataset_template_strategy = "generic"
        super().__init__(cfg)
        if tokenizer is not None:
            self.setup_vocab_expansion(tokenizer)

    @staticmethod
    def _flatten_caption_array(captions) -> List[str]:
        flattened = []
        for item in captions:
            if hasattr(item, "reshape"):
                item = item.reshape(-1)[0]
            elif isinstance(item, (list, tuple)):
                item = item[0]
            flattened.append(str(item).strip())
        return flattened

    def _resolve_template_strategy(self, dataset_root: str, ve_cfg: Dict) -> str:
        explicit = ve_cfg.get("dataset_strategy")
        if explicit:
            strategy = str(explicit).strip().lower()
            if strategy != "ettm1":
                raise NotImplementedError(
                    f"Only dataset_strategy='ETTm1' is supported for now, got '{explicit}'."
                )
            return "ettm1"
        dataset_name = os.path.basename(os.path.normpath(dataset_root)).lower()
        if dataset_name == "ettm1":
            return "ettm1"
        raise NotImplementedError(
            "Stage1LatentCaptionModelVE currently supports only ETTm1. "
            "Please set vocab_expansion.dataset_strategy='ETTm1' or point data.dataset_root to ETTm1."
        )

    @classmethod
    def _normalize_template_ettm1_base(cls, sentence: str) -> str:
        s = " ".join(sentence.replace("\n", " ").split())
        if s.startswith("This sequence is "):
            return "This sequence is <VAR_NAME>."
        if s.startswith("The main season cycles is around "):
            return "The main season cycles is around <SEASON_CYCLE>."
        if s.startswith("For the ovearll shape,"):
            return "For the ovearll shape, <OVERALL_TREND>."
        if s.startswith("The distribution of the value in time series is "):
            return "The distribution of the value in time series is <SKEWNESS_KURTOSIS>."
        return cls.NUMBER_PATTERN.sub("<NUM>", s)

    @classmethod
    def _normalize_template_ettm1(cls, sentence: str) -> str:
        s = " ".join(sentence.replace("\n", " ").split())
        local = cls.LOCAL_SEGMENT_PATTERN.match(s)
        if local is not None:
            segment = local.group(1).lower()
            clause = local.group(2).strip()
            if cls.PEAK_PREFIX_PATTERN.match(clause):
                rest = cls.PEAK_PREFIX_PATTERN.sub("", clause, count=1).strip()
                rest = cls.NUMBER_PATTERN.sub("<NUM>", rest)
                return f"At the {segment}: there are <NUM> peaks, {rest}."
            clause = cls.NUMBER_PATTERN.sub("<NUM>", clause)
            return f"At the {segment}: {clause}."
        return cls._normalize_template_ettm1_base(s)

    def _normalize_template(self, sentence: str) -> str:
        if self._dataset_template_strategy == "ettm1":
            return self._normalize_template_ettm1(sentence)
        raise NotImplementedError(
            f"Unsupported dataset_strategy='{self._dataset_template_strategy}'. "
            "Please add a dataset-specific template normalizer in Stage1LatentCaptionModelVE."
        )

    def _extract_templates(self, caption: str) -> List[str]:
        text = caption.replace("\n", " ")
        templates = []
        for sentence in self.SENTENCE_PATTERN.findall(text):
            norm = self._normalize_template(sentence)
            if norm:
                templates.append(norm)
        return templates

    def _mine_templates_from_training_set(self, dataset_root: str, min_count: int, max_tokens: int) -> List[Dict]:
        caps_path = os.path.join(dataset_root, "train_text_caps.npy")
        if not os.path.exists(caps_path):
            raise FileNotFoundError(f"Missing train captions file for VE: {caps_path}")
        caps = np.load(caps_path, allow_pickle=True)
        captions = self._flatten_caption_array(caps)
        counter = Counter()
        for caption in captions:
            counter.update(self._extract_templates(caption))
        rows = []
        for template, count in counter.most_common():
            if count < min_count:
                continue
            rows.append({"template": template, "count": int(count)})
            if len(rows) >= max_tokens:
                break
        return rows

    @staticmethod
    def _mean_init_embedding(embedding_table: torch.Tensor, token_ids: List[int]) -> torch.Tensor:
        if not token_ids:
            return embedding_table.mean(dim=0)
        ids = torch.tensor(token_ids, dtype=torch.long, device=embedding_table.device)
        return embedding_table.index_select(0, ids).mean(dim=0)

    def _initialize_new_token_embeddings(self, tokenizer, token_to_template: List[Dict], old_vocab_size: int) -> None:
        input_embed = self.llm.get_input_embeddings()
        output_embed = self.llm.get_output_embeddings()

        with torch.no_grad():
            input_weight = input_embed.weight
            for row in token_to_template:
                token = row["token"]
                template = row["template"]
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id < old_vocab_size:
                    continue
                source_ids = tokenizer(template, add_special_tokens=False)["input_ids"]
                vec = self._mean_init_embedding(input_weight, source_ids)
                input_weight[token_id].copy_(vec)

            if output_embed is not None and output_embed.weight is not input_weight:
                output_weight = output_embed.weight
                for row in token_to_template:
                    token = row["token"]
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    if token_id < old_vocab_size:
                        continue
                    output_weight[token_id].copy_(input_weight[token_id])

    def _register_embedding_grad_mask(self, old_vocab_size: int, new_vocab_size: int) -> None:
        if new_vocab_size <= old_vocab_size:
            return
        row_mask = torch.zeros(new_vocab_size, dtype=torch.float32)
        row_mask[old_vocab_size:new_vocab_size] = 1.0
        self.register_buffer("_ve_embedding_row_mask", row_mask, persistent=False)

        input_weight = self.llm.get_input_embeddings().weight
        input_weight.requires_grad = True

        def mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self._ve_embedding_row_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
            return grad * mask

        self._ve_grad_hooks.append(input_weight.register_hook(mask_grad))

        output_embed = self.llm.get_output_embeddings()
        if output_embed is not None and output_embed.weight is not input_weight:
            output_embed.weight.requires_grad = True
            self._ve_grad_hooks.append(output_embed.weight.register_hook(mask_grad))

    def setup_vocab_expansion(self, tokenizer) -> None:
        ve_cfg = dict(self.cfg.get("vocab_expansion", {}))
        if not ve_cfg:
            ve_cfg = dict(self.cfg.get("model", {}).get("vocab_expansion", {}))
        enabled = bool(ve_cfg.get("enabled", True))
        if not enabled:
            return

        data_cfg = self.cfg.get("data", {})
        dataset_root = data_cfg.get("dataset_root")
        if dataset_root is None:
            raise ValueError("Stage1LatentCaptionModelVE requires cfg.data.dataset_root for template mining.")

        min_count = int(ve_cfg.get("min_count", 20))
        max_tokens = int(ve_cfg.get("max_tokens", 256))
        token_prefix = ve_cfg.get("token_prefix", "<CAPTPL_")
        token_suffix = ve_cfg.get("token_suffix", ">")
        self._dataset_template_strategy = self._resolve_template_strategy(dataset_root=dataset_root, ve_cfg=ve_cfg)

        templates = self._mine_templates_from_training_set(
            dataset_root=dataset_root,
            min_count=min_count,
            max_tokens=max_tokens,
        )

        special_tokens = []
        token_to_template = []
        for idx, row in enumerate(templates, start=1):
            token = f"{token_prefix}{idx:03d}{token_suffix}"
            special_tokens.append(token)
            token_to_template.append(
                {
                    "token": token,
                    "template": row["template"],
                    "count": row["count"],
                }
            )
            self._template_to_token[row["template"]] = token

        old_vocab_size = len(tokenizer)
        added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if added > 0:
            self.llm.resize_token_embeddings(len(tokenizer))
        new_vocab_size = len(tokenizer)
        self._initialize_new_token_embeddings(tokenizer, token_to_template, old_vocab_size)
        self._register_embedding_grad_mask(old_vocab_size, new_vocab_size)

        self.vocab_expansion_summary = {
            "enabled": True,
            "dataset_root": os.path.abspath(dataset_root),
            "dataset_template_strategy": self._dataset_template_strategy,
            "min_count": min_count,
            "max_tokens": max_tokens,
            "requested_tokens": len(special_tokens),
            "num_added_tokens": int(added),
            "old_vocab_size": int(old_vocab_size),
            "new_vocab_size": int(new_vocab_size),
            "token_to_template": token_to_template,
        }
        # LoRA leaves base embedding frozen; enable embedding grads for newly added rows.
        self._base_requires_grad = {
            name: param.requires_grad for name, param in self.named_parameters()
        }

    def encode_caption_with_special_tokens(self, caption: str) -> str:
        if not self._template_to_token:
            return caption
        output_parts = []
        for sentence in self.SENTENCE_PATTERN.findall(caption.replace("\n", " ")):
            norm = self._normalize_template(sentence)
            token = self._template_to_token.get(norm)
            if token is not None:
                output_parts.append(token)
            else:
                output_parts.append(" ".join(sentence.split()))
        if not output_parts:
            return caption
        return " ".join(output_parts)

    def save_vocab_expansion_artifacts(self, output_dir: str) -> Optional[str]:
        if not self.vocab_expansion_summary.get("enabled", False):
            return None
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "vocab_expansion_summary.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab_expansion_summary, f, ensure_ascii=False, indent=2)
        return save_path
