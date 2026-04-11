from typing import Optional, Union
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLPreTrainedModel, Cache
from transformers import Qwen3VLVisionModel, Qwen3VLTextModel
from transformers import Blip2QFormerConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast, Qwen3VLVisionRotaryEmbedding, Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils import is_torchdynamo_compiling
from transformers import Qwen3VLTextConfig
from transformers import Qwen3VLProcessor, ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs
# from transformers.utils import logging
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from PIL import Image
import numpy as np
from qwen_vl_utils import process_vision_info



def build_template_inputs(
    processor,
    batch_size,
    device: str = "cpu",
):

    # prompt_text = r"""
    # You are a scientific time-series analyst.
    #
    # You are given an image of a 1D time series plot (x-axis is time index, y-axis is signal value).
    #
    # Your task is to describe the overall morphology (global shape pattern) of the time series.
    #
    # IMPORTANT INSTRUCTIONS:
    # - Do NOT provide any specific numerical values.
    # - Do NOT mention exact time indices or ranges.
    # - Do NOT segment the time series into numbered sections.
    # - Do NOT use precise quantities (no decimals, no exact amplitudes).
    # - Use qualitative descriptions only (e.g., gradual rise, sharp drop, high volatility, multi-phase evolution, oscillatory behavior).
    # - Focus on the overall structure, trend transitions, symmetry/asymmetry, volatility level, and presence of peaks or troughs.
    #
    # ### Output format (must follow exactly):
    #
    # **Overall morphology:** <One coherent paragraph describing the global shape in qualitative terms only.>
    #
    # Now generate the description.
    # """.strip()

    min_pixels = 0
    max_pixels = 64 * 14 * 14

    prompt_text = r"""
    You are a scientific time-series analyst.

    The image shows time-series plots for ONE channel.

    The time series is divided into EXACTLY 4 segments.
    Each segment is displayed as one subplot.

    Your task is to describe the morphology of each segment separately.

    Instructions:
    - There are exactly 4 segments: Segment 1, Segment 2, Segment 3, Segment 4.
    - Write exactly ONE sentence for each segment.
    - Use only qualitative descriptions (e.g., increasing trend, decreasing trend, oscillatory behavior, sudden spike, high volatility, stable plateau).
    - Do NOT mention numerical values.
    - Do NOT mention time indices.
    - Do NOT create additional segments.
    - Only describe Segment 1 to Segment 4.

    Output format (strictly follow this format):

    Segment 1: <one sentence description>
    Segment 2: <one sentence description>
    Segment 3: <one sentence description>
    Segment 4: <one sentence description>
    """.strip()

    # -------------------------
    # 1) Dummy image placeholder
    # -------------------------
    # dummy_arr = np.zeros((400, 100, 3), dtype=np.uint8)
    # dummy_img = Image.fromarray(dummy_arr)
    dummy_img = "./dummy_image.png"
    # -------------------------
    # 2) Prompt-only messages (for prompt_len)
    # -------------------------

    prompt_messages = []

    for _ in range(batch_size):
        prompt_messages.append(
            [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": dummy_img,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }]
        )

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in prompt_messages
    ]

    prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_messages)

    prompt_inputs = processor(
        text=texts,
        images=prompt_image_inputs,
        videos=prompt_video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return prompt_inputs



def get_narrow_latent(model, hidden_states, grid_thw, **kwargs):
    hidden_states = model.patch_embed(hidden_states)

    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = model.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    # print(f"seq_len: {seq_len}")

    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    # print(f"grid_thw.shape={grid_thw.shape}")
    # print(f"position_embeddings.shape: {position_embeddings[0].shape}; position_embeddings.shape: {position_embeddings[1].shape}")
    # print(f"cu_seqlens.shape: {cu_seqlens.shape}")
    # print(f"hidden_states.shape: {hidden_states.shape}")

    for layer_num, blk in enumerate(model.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in model.deepstack_visual_indexes:
            # deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
            #     hidden_states
            # )
            deepstack_feature_lists.append(hidden_states)

    hidden_states = model.merger(hidden_states)

    return hidden_states, deepstack_feature_lists



def generated_latent_feed_visual_encoder(
        model: Qwen3VLVisionModel,
        hidden_states,
        shallow_visual_latent,
        grid_thw,
        **kwargs
):
    hidden_states = model.patch_embed(hidden_states)
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)


    # print(f"position_embeddings.shape: {position_embeddings[0].shape}; position_embeddings.shape: {position_embeddings[1].shape}")
    # print(f"cu_seqlens.shape: {cu_seqlens.shape}")


    deepstack_feature_lists = []
    hidden_states = shallow_visual_latent.contiguous()
    # print(f"hidden_states.shape: {hidden_states.shape}")
    # append the first deepstack feature
    deepstack_feature = model.deepstack_merger_list[0](hidden_states)
    deepstack_feature_lists.append(deepstack_feature)

    for layer_num, blk in enumerate(model.blocks):
        if layer_num <= model.deepstack_visual_indexes[0]:
            continue
        # breakpoint()
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in model.deepstack_visual_indexes:
            deepstack_feature = model.deepstack_merger_list[model.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = model.merger(hidden_states)

    return hidden_states, deepstack_feature_lists

    # deepstack_feature_lists = []
    # for layer_num, blk in enumerate(model.blocks):
    #     hidden_states = blk(
    #         hidden_states,
    #         cu_seqlens=cu_seqlens,
    #         position_embeddings=position_embeddings,
    #         **kwargs,
    #     )
    #     if layer_num in model.deepstack_visual_indexes:
    #         deepstack_feature = model.deepstack_merger_list[model.deepstack_visual_indexes.index(layer_num)](
    #             hidden_states
    #         )
    #         deepstack_feature_lists.append(deepstack_feature)
    #         # breakpoint()
    #
    # hidden_states = model.merger(hidden_states)
    #
    # return hidden_states, deepstack_feature_lists



# def generated_latent_feed_visual_encoder(
#         model,
#         hidden_states: torch.Tensor,
#         shallow_visual_latent: torch.Tensor,
#         grid_thw: torch.Tensor, **kwargs
# ) -> torch.Tensor:
#     """
#     Args:
#         hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
#             The final hidden states of the model.
#         grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
#             The temporal, height and width of feature shape of each image in LLM.
#
#     Returns:
#         `torch.Tensor`: hidden_states.
#     """
#     hidden_states = model.patch_embed(hidden_states)
#
#     pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
#     hidden_states = hidden_states + pos_embeds
#
#     rotary_pos_emb = model.rot_pos_emb(grid_thw)
#
#     seq_len, _ = hidden_states.size()
#     hidden_states = hidden_states.reshape(seq_len, -1)
#     rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
#     emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
#     position_embeddings = (emb.cos(), emb.sin())
#
#     cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
#         dim=0,
#         # Select dtype based on the following factors:
#         #  - FA2 requires that cu_seqlens_q must have dtype int32
#         #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
#         # See https://github.com/huggingface/transformers/pull/34852 for more information
#         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
#     )
#     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
#
#     deepstack_feature_lists = []
#     for layer_num, blk in enumerate(model.blocks):
#         hidden_states = blk(
#             hidden_states,
#             cu_seqlens=cu_seqlens,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         if layer_num in model.deepstack_visual_indexes:
#             deepstack_feature = model.deepstack_merger_list[model.deepstack_visual_indexes.index(layer_num)](
#                 hidden_states
#             )
#             deepstack_feature_lists.append(deepstack_feature)
#
#     hidden_states = model.merger(hidden_states)
#
#     return hidden_states, deepstack_feature_lists





class Qwen3VisionEncoder(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__()

        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,   # 先别放GPU，避免爆显存
        )

        self.visual = full_model.model.visual   # 只拿 vision encoder
        del full_model                          # 删除 text + lm_head

        # self.visual.to(device)
        # self.visual.eval()

    # @torch.no_grad()
    # def encode_images(self, pixel_values, image_grid_thw):
    #
    #     image_embeds, deepstack_embeds = get_narrow_latent(self.visual, pixel_values, grid_thw=image_grid_thw)
    #     # image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    #
    #     # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    #     # image_embeds = torch.split(image_embeds, split_sizes)
    #
    #     # image_embeds = torch.stack(image_embeds, dim=0)  # (B, L, D)
    #
    #     # merge = self.visual.spatial_merge_size
    #     # H = image_grid_thw[0, 1].item() // merge
    #     # W = image_grid_thw[0, 2].item() // merge
    #
    #     # B, L, D = image_embeds.shape
    #     # image_embeds = image_embeds.reshape(B, H, W, D).permute(0, 3, 1, 2)
    #
    #     latent = deepstack_embeds[0]
    #     breakpoint()
    #     return latent   # (B, D, H, W)

    @torch.no_grad()
    def encode_images(self, pixel_values, image_grid_thw):
        image_embeds, deepstack_embeds = get_narrow_latent(self.visual, pixel_values, grid_thw=image_grid_thw)
        # image_embeds, deepstack_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)

        image_embeds = torch.stack(image_embeds, dim=0)  # (B, L, D)

        merge = self.visual.spatial_merge_size
        H = image_grid_thw[0, 1].item() // merge
        W = image_grid_thw[0, 2].item() // merge

        B, L, D = image_embeds.shape
        image_embeds = image_embeds.reshape(B, H, W, D).permute(0, 3, 1, 2)

        latent = deepstack_embeds[0].reshape(B, H * 2, W * 2, 1152).permute(0, 3, 1, 2)
        return latent  # (B, D, H, W)


class Stage1_Qwen3VLModel(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            shallow_visual_latent = kwargs.get("shallow_visual_latent", None)

            if shallow_visual_latent is not None:
                image_embeds, deepstack_image_embeds = self.get_image_features_from_generated_latent(
                    shallow_visual_latent, pixel_values, image_grid_thw
                )
            else:
                raise ValueError("You must specify shallow_visual_latent")
                image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            # image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)

            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


    def get_image_features_from_generated_latent(
            self,
            shallow_visual_latent: torch.FloatTensor,
            hidden_states: torch.FloatTensor,
            image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """

        shallow_visual_latent = shallow_visual_latent.type(self.visual.dtype)
        hidden_states = hidden_states.type(self.visual.dtype)
        # image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

        ######## START: go through the visual encoder  ########
        image_embeds, deepstack_image_embeds = generated_latent_feed_visual_encoder(
            self.visual,
            hidden_states,
            shallow_visual_latent,
            image_grid_thw
        )
        ######## END: go through the visual encoder  ########

        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds



class Stage1_Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super(Qwen3VLPreTrainedModel, self).__init__(config)
        self.model = Stage1_Qwen3VLModel(config)
        # self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()


    def forward(
            self,
            shallow_visual_latent=None,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs = self.model(
            shallow_visual_latent=shallow_visual_latent,
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )



    # def prepare_inputs_for_generation(
    #     self,
    #     input_ids,
    #     past_key_values=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     cache_position=None,
    #     position_ids=None,
    #     use_cache=True,
    #     pixel_values=None,
    #     pixel_values_videos=None,
    #     image_grid_thw=None,
    #     video_grid_thw=None,
    #     **kwargs,
    # ):
    #     # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
    #
    #     model_inputs = super().prepare_inputs_for_generation(
    #         input_ids,
    #         past_key_values=past_key_values,
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         cache_position=cache_position,
    #         position_ids=position_ids,
    #         pixel_values=pixel_values,
    #         pixel_values_videos=pixel_values_videos,
    #         image_grid_thw=image_grid_thw,
    #         video_grid_thw=video_grid_thw,
    #         use_cache=use_cache,
    #         **kwargs,
    #     )
    #
    #     # Qwen3VL position_ids are prepareed with rope_deltas in forward
    #     model_inputs["position_ids"] = None
    #
    #     if cache_position[0] != 0:
    #         model_inputs["pixel_values"] = None
    #         model_inputs["pixel_values_videos"] = None
    #
    #     if "image_embeds" in kwargs and kwargs["image_embeds"] is not None:
    #         model_inputs["image_embeds"] = kwargs["image_embeds"]
    #
    #     return model_inputs


    def decode(self, latents, tokens_per_sample):

        batch_size = latents.shape[0] // tokens_per_sample
        processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy_batch = build_template_inputs(
            processor=processor,
            batch_size=batch_size,
            device=device,
        )

        # dummy_batch.update({"image_embeds": latents})
        generated_ids = self.generate(
            **dummy_batch,
            shallow_visual_latent=latents,
            max_new_tokens=1024,
            do_sample=False,
        )

        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(dummy_batch["input_ids"], generated_ids)
            ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        # for out in output_text:
        #     print(out)

        return output_text


import torch
import numpy as np
from PIL import Image


def pixel_values_to_pil(pixel_values, image_processor, index=0):
    """
    Convert Qwen3VLProcessor output pixel_values back to a PIL image
    (approximately, after preprocessing).

    Args:
        pixel_values: torch.Tensor or np.ndarray
            Shape can be (B, C, H, W) or (C, H, W)
        image_processor:
            Qwen2VLImageProcessor or similar, must contain mean/std info.
        index: int
            which image in batch to convert.

    Returns:
        PIL.Image.Image
    """

    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)

    # handle batch
    if pixel_values.dim() == 4:
        x = pixel_values[index]   # (C, H, W)
    elif pixel_values.dim() == 3:
        x = pixel_values
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

    # get mean/std
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)

    if mean is None or std is None:
        raise ValueError("image_processor must have image_mean and image_std")

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # de-normalize
    x = x * std + mean

    # clamp to [0, 1]
    x = torch.clamp(x, 0.0, 1.0)

    # CHW -> HWC
    x = x.permute(1, 2, 0)

    # float -> uint8
    x = (x * 255.0).round().to(torch.uint8).cpu().numpy()

    return Image.fromarray(x)



def test_same_text_output(latents):
    # model_name = "/Users/zhc/Downloads/Qwen3-VL-8B-Instruct"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Stage1_Qwen3VLForConditionalGeneration.from_pretrained(model_name).eval().to(device)

    if latents is not None:
        latents = latents.to(device)
        # breakpoint()
    else:
        latents = torch.randn(784, 1152).to(model.device)
    # latents = torch.load("image_embeds_example.pt").to(model.device).reshape(2, 196, 4096)
    # latents = example_inputs['shallow_latent'].to(device).reshape(1, 1152, -1).permute(0, 2, 1).squeeze(0)
    image_size = 450
    with torch.no_grad():
        model.decode(latents, image_size)


def test_narrow_latent():
    # model_name = "/Users/zhc/Downloads/Qwen3-VL-8B-Instruct"
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Qwen3VLProcessor.from_pretrained(model_name)
    model = Qwen3VisionEncoder(model_name).eval().to(device)
    dummy_batch = build_template_inputs(
        processor=processor,
        batch_size=2,
        device=device,
    )

    latents = model.encode_images(dummy_batch["pixel_values"].to(device), dummy_batch["image_grid_thw"].to(device))
    # print(latents.shape)
    return latents
    # latents = torch.load("image_embeds_example.pt").to(model.device).reshape(2, 196, 4096)
    # latents = torch.randn(2, 784, 1152).to(model.device)
    # latents = example_inputs['shallow_latent'].to(device).reshape(1, 1152, -1).permute(0, 2, 1).squeeze(0)
    # image_size = 450
    # with torch.no_grad():
    #     model.decode(latents, image_size)


if __name__ == "__main__":
    latents = test_narrow_latent()
    test_same_text_output(latents)
    test_same_text_output(None)


