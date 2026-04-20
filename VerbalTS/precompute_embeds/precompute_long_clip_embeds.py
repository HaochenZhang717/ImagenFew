import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np


# =========================
# CLIP Text Encoder
# =========================
class ClipTextEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        model_path = "/playpen-shared/haochenz/long_clip"

        self.model = CLIPTextModelWithProjection.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 正确 max length
        self.max_length = min(
            self.model.config.max_position_embeddings,
            self.tokenizer.model_max_length
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list,
            padding="max_length",  # ✅ 关键！
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_emb = outputs.last_hidden_state  # (B, L, D)

        return text_emb, attention_mask



def precompute_from_npy(
    caps_path,
    save_path,
    npy_name,
    split="train",
    batch_size=64,
    device="cuda"
):

    print("Loading captions...")
    caps = np.load(f"{caps_path}/{split}_{npy_name}.npy", allow_pickle=True)
    encoder = ClipTextEncoder(device).to(device)

    all_embeds = []
    all_masks = []
    # =========================
    # batch encode
    # =========================
    for i in tqdm(range(0, len(caps), batch_size)):
        batch_text = [cap[0] for cap in caps[i:i + batch_size]]
        # batch_text = caps[i:i + batch_size]
        embeds, attn_masks = encoder(batch_text)  # (B, L, D)
        all_embeds.append(embeds.cpu())
        all_masks.append(attn_masks.cpu())

    # =========================
    # 拼成 (N, L, D)
    # =========================
    all_embeds = torch.cat(all_embeds, dim=0)  # (N, L, D)
    all_masks = torch.cat(all_masks, dim=0)
    # =========================
    # 保存
    # =========================
    # torch.save(all_embeds, save_path)

    torch.save({
        "embeddings": all_embeds,  # (N, L, D)
        "all_masks": all_masks
    }, save_path)
    print(f"Saved to {save_path}")
    print("Shape:", all_embeds.shape)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--caps_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--npy_name", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    precompute_from_npy(
        caps_path=args.caps_path,
        save_path=args.save_path,
        npy_name=args.npy_name,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )