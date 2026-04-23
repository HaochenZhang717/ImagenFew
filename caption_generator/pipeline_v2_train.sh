HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"



#python prepare_pipeline_v2_dataset.py --config configs/synthetic_u_pipeline_v2_qwen3vl.yaml
#python prepare_pipeline_v2_dataset_segments.py --config configs/synthetic_u_pipeline_v2_qwen3vl_segments.yaml

# test render image

#python - <<'PY'
#import os
#import numpy as np
#
#dataset_root = "../data/VerbalTSDatasets/synthetic_u"
#image_root = os.path.join(dataset_root, "pipeline_v2_images_100")
#
#for split in ["train", "valid", "test"]:
#    ts_path = os.path.join(dataset_root, f"{split}_ts.npy")
#    img_dir = os.path.join(image_root, split)
#
#    ts_count = len(np.load(ts_path, allow_pickle=True))
#    png_count = len([f for f in os.listdir(img_dir) if f.endswith(".png")]) if os.path.isdir(img_dir) else -1
#
#    print(f"{split}: ts={ts_count}, png={png_count}, match={ts_count == png_count}")
#PY



CUDA_VISIBLE_DEVICES=3 python train_pipeline_v2.py --config configs/ettm1_pipeline_v2_qwen3vl.yaml