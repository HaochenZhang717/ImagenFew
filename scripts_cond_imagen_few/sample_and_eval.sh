export CUDA_VISIBLE_DEVICES=0

python /playpen-shared/haochenz/ImagenFew/scripts_cond_imagen_few/sample_posterior_imagenfew_cross_attention.py \
  --config /playpen-shared/haochenz/ImagenFew/configs/conditional_imagen_few/ETTh2.yaml \
  --model-ckpt /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/ETTh2/19c5b45b-e429-4f16-96dc-180ad93dd2df/ImagenFewCrossAttention.pt \
  --time-series-path /playpen-shared/haochenz/ImagenFew/logs/ImagenFew/ETTh2/404bb2bf-e4b2-412c-87b5-032bac01e4f9/generated_ETTh2_train.npy \
  --output /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/ETTh2/19c5b45b-e429-4f16-96dc-180ad93dd2df/posterior_samples.npy \
  --split train \
  --eval-metrics "disc contextFID"


python /playpen-shared/haochenz/ImagenFew/scripts_cond_imagen_few/sample_posterior_imagenfew_cross_attention.py \
  --config /playpen-shared/haochenz/ImagenFew/configs/conditional_imagen_few/AirQuality.yaml \
  --model-ckpt /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/AirQuality/62c04c06-809c-4db6-8fb0-001d6f46072b/ImagenFewCrossAttention.pt \
  --time-series-path /playpen-shared/haochenz/ImagenFew/logs/ImagenFew/AirQuality/d040753f-6f2d-4c58-bae3-ff2764e47492/generated_AirQuality_train.npy \
  --output /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/AirQuality/62c04c06-809c-4db6-8fb0-001d6f46072b/posterior_samples.npy \
  --split train \
  --eval-metrics "disc contextFID"



python /playpen-shared/haochenz/ImagenFew/scripts_cond_imagen_few/sample_posterior_imagenfew_cross_attention.py \
  --config /playpen-shared/haochenz/ImagenFew/configs/conditional_imagen_few/mujoco.yaml \
  --model-ckpt /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/mujoco/7a5fde19-f926-4ab5-82fb-98272fd4c5d5/ImagenFewCrossAttention.pt \
  --time-series-path /playpen-shared/haochenz/ImagenFew/logs/ImagenFew/Mujoco/faed7452-4f59-41dd-9bb9-882467dcb5b0/generated_mujoco_train.npy \
  --output /playpen-shared/haochenz/ImagenFew/logs/CondImagenFewFinetune/mujoco/7a5fde19-f926-4ab5-82fb-98272fd4c5d5/posterior_samples.npy \
  --split train \
  --eval-metrics "disc contextFID"
