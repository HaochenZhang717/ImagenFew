from types import SimpleNamespace
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _resolve_torch_dtype(dtype_name: Optional[str]):
    if dtype_name is None or dtype_name == "auto":
        return None
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


class StaticSoftPrompt(nn.Module):
    def __init__(self, prompt_length: int, hidden_size: int, init_std: float = 0.02) -> None:
        super().__init__()
        prompt = torch.empty(prompt_length, hidden_size)
        nn.init.normal_(prompt, mean=0.0, std=init_std)
        self.prompt = nn.Parameter(prompt)

    def forward(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)


class PipelineV2CaptionModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.model_name = model_cfg["vlm_name"]
        torch_dtype = _resolve_torch_dtype(model_cfg.get("torch_dtype"))

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
        )
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            low_cpu_mem_usage=model_cfg.get("low_cpu_mem_usage", True),
        )
        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad = False

        hidden_size = self.vlm.get_input_embeddings().embedding_dim
        self.soft_prompt = StaticSoftPrompt(
            prompt_length=int(model_cfg.get("soft_prompt_tokens", 16)),
            hidden_size=hidden_size,
            init_std=float(model_cfg.get("soft_prompt_init_std", 0.02)),
        )
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.processor.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id.")
            self.pad_token_id = self.processor.tokenizer.eos_token_id

    def train(self, mode: bool = True):
        super().train(mode)
        # The spec freezes the VLM completely, so keep it in eval mode even while
        # optimizing the soft prompt.
        self.vlm.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.soft_prompt.parameters()).device

    @property
    def soft_prompt_tokens(self) -> int:
        return int(self.soft_prompt.prompt.shape[0])

    def trainable_parameters(self):
        return self.soft_prompt.parameters()

    def _compose_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        image_embeds, deepstack_image_embeds = self.vlm.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        image_mask, _ = self.vlm.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        prompt_embeds = self.soft_prompt(
            batch_size=input_ids.shape[0],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        full_inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        prompt_attention = attention_mask.new_ones((attention_mask.shape[0], self.soft_prompt_tokens))
        full_attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        dummy_prompt_ids = input_ids.new_full((input_ids.shape[0], self.soft_prompt_tokens), self.pad_token_id)
        full_input_ids = torch.cat([dummy_prompt_ids, input_ids], dim=1)

        full_labels = None
        if labels is not None:
            prompt_labels = labels.new_full((labels.shape[0], self.soft_prompt_tokens), -100)
            full_labels = torch.cat([prompt_labels, labels], dim=1)

        full_visual_pos_masks = torch.cat(
            [
                visual_pos_masks.new_zeros((visual_pos_masks.shape[0], self.soft_prompt_tokens)),
                visual_pos_masks,
            ],
            dim=1,
        )

        position_ids, rope_deltas = self.vlm.model.get_rope_index(
            full_input_ids,
            image_grid_thw,
            None,
            attention_mask=full_attention_mask,
        )

        return {
            "inputs_embeds": full_inputs_embeds,
            "attention_mask": full_attention_mask,
            "labels": full_labels,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "visual_pos_masks": full_visual_pos_masks,
            "deepstack_visual_embeds": deepstack_image_embeds,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        assembled = self._compose_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )
        self.vlm.model.rope_deltas = assembled["rope_deltas"]
        outputs = self.vlm.model.language_model(
            input_ids=None,
            position_ids=assembled["position_ids"],
            attention_mask=assembled["attention_mask"],
            inputs_embeds=assembled["inputs_embeds"],
            visual_pos_masks=assembled["visual_pos_masks"],
            deepstack_visual_embeds=assembled["deepstack_visual_embeds"],
            use_cache=False,
        )
        logits = self.vlm.lm_head(outputs.last_hidden_state)
        loss = None
        if assembled["labels"] is not None:
            loss = self.vlm.loss_function(
                logits=logits,
                labels=assembled["labels"],
                vocab_size=self.vlm.config.text_config.vocab_size,
            )
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        eos_token_id = eos_token_id if eos_token_id is not None else self.processor.tokenizer.eos_token_id
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
            )
            next_token_logits = outputs.logits[:, -1, :]
            if temperature and temperature > 0:
                scaled_logits = next_token_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    filtered_logits = scaled_logits.clone()
                    filtered_logits.scatter_(
                        1,
                        sorted_indices,
                        torch.where(
                            sorted_indices_to_remove,
                            torch.full_like(sorted_logits, float("-inf")),
                            sorted_logits,
                        ),
                    )
                    probs = torch.softmax(filtered_logits, dim=-1)
                else:
                    probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, cur_attention_mask.new_ones((cur_attention_mask.shape[0], 1))],
                dim=1,
            )
            if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
                break

        return cur_input_ids
