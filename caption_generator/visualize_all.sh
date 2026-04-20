
python scripts/visualize_caption_embedding_distribution.py \
  --generated-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/synthetic_m/generated_text_caps.npy \
  --real-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/synthetic_m/test_text_caps.npy \
  --output-dir ./visuals/synthetic_m \
  --device cuda \
  --batch-size 8



python scripts/visualize_caption_embedding_distribution.py \
  --generated-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/istanbul_traffic/generated_text_caps.npy \
  --real-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/istanbul_traffic/test_text_caps.npy \
  --output-dir ./visuals/istanbul_traffic \
  --device cuda \
  --batch-size 8


python scripts/visualize_caption_embedding_distribution.py \
  --generated-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/ETTm1/generated_text_caps.npy \
  --real-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/ETTm1/test_text_caps.npy \
  --output-dir ./visuals/ETTm1 \
  --device cuda \
  --batch-size 8


#python scripts/visualize_caption_embedding_distribution.py \
#  --generated-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/synthetic_u/generated_text_caps.npy \
#  --real-path /playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/synthetic_u/test_text_caps.npy \
#  --output-dir ./visuals/synthetic_u \
#  --device cuda \
#  --batch-size 8

