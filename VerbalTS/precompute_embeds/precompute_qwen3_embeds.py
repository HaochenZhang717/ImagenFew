import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


class QwenTextEncoder(torch.nn.Module):
    def __init__(
        self,
        device,
        model_name="Qwen/Qwen3-Embedding-4B",
        max_length=8192,
        use_instruct=False,
        instruct_text="Represent the time-series caption for retrieval and generation conditioning.",
        use_flash_attn=False,
        use_fp16=False,
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        self.use_instruct = use_instruct
        self.instruct_text = instruct_text

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        model_kwargs = {}
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if use_fp16 and str(device).startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, text_list):
        if self.use_instruct:
            text_list = [get_detailed_instruct(self.instruct_text, text) for text in text_list]

        batch_dict = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings, batch_dict["attention_mask"]


def load_captions(caps_path, split, npy_name):
    npy_path = os.path.join(caps_path, f"{split}_{npy_name}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Caption file not found: {npy_path}")
    caps = np.load(npy_path, allow_pickle=True)
    text_list = []
    for cap in caps:
        if isinstance(cap, np.ndarray) and cap.ndim > 0:
            text_list.append(str(cap[0]))
        elif isinstance(cap, (list, tuple)):
            text_list.append(str(cap[0]))
        else:
            text_list.append(str(cap))
    return text_list


def precompute_from_npy(
    caps_path,
    save_path,
    npy_name,
    split="train",
    batch_size=16,
    device="cuda",
    model_name="Qwen/Qwen3-Embedding-4B",
    max_length=8192,
    use_instruct=False,
    instruct_text="Represent the time-series caption for retrieval and generation conditioning.",
    use_flash_attn=False,
    use_fp16=False,
):
    print("Loading captions...")
    caps = load_captions(caps_path, split, npy_name)
    encoder = QwenTextEncoder(
        device=device,
        model_name=model_name,
        max_length=max_length,
        use_instruct=use_instruct,
        instruct_text=instruct_text,
        use_flash_attn=use_flash_attn,
        use_fp16=use_fp16,
    )

    all_embeds = []
    for i in tqdm(range(0, len(caps), batch_size)):
        batch_text = caps[i : i + batch_size]
        embeds, _ = encoder(batch_text)
        all_embeds.append(embeds.cpu())

    all_embeds = torch.cat(all_embeds, dim=0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(all_embeds, save_path)
    print(f"Saved to {save_path}")
    print("Embedding shape:", all_embeds.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--npy_name", type=str, default="text_caps")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--use_instruct", action="store_true")
    parser.add_argument(
        "--instruct_text",
        type=str,
        default="Represent the time-series caption for retrieval and generation conditioning.",
    )
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    precompute_from_npy(
        caps_path=args.caps_path,
        save_path=args.save_path,
        npy_name=args.npy_name,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model_name,
        max_length=args.max_length,
        use_instruct=args.use_instruct,
        instruct_text=args.instruct_text,
        use_flash_attn=args.use_flash_attn,
        use_fp16=args.use_fp16,
    )
