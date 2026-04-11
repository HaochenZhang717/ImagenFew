import torch
import re


def build_timeseries_latents_fast(
    embedding_dict,
    num_images,
    num_seg,
    num_ch,
):
    """
    embedding_dict:
        {image_name: tensor(H,W,D)}

    return:
        tensor (N, S, C, H, W, D)
    """

    # sample shape
    sample = next(iter(embedding_dict.values()))
    H, W, D = sample.shape

    latents = torch.zeros(
        num_images,
        num_seg,
        num_ch,
        H,
        W,
        D,
        dtype=sample.dtype,
    )

    pattern = re.compile(r'image(\d+)_seg(\d+)_ch(\d+)')

    for name, emb in embedding_dict.items():

        m = pattern.match(name)
        if m is None:
            continue

        image_id = int(m.group(1))
        seg_id = int(m.group(2))
        ch_id = int(m.group(3))

        latents[image_id, seg_id, ch_id] = emb

    return latents


if __name__ == "__main__":
    embedding_dict = torch.load("/playpen/haochenz/LitsDatasets/128_len_vision_latent_raw/synth_m/train.pt")
    latents = build_timeseries_latents_fast(
        embedding_dict,
        num_images=24000,
        num_seg=4,
        num_ch=2,
    )
    print(latents.shape)


