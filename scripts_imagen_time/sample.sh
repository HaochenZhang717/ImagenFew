python scripts_imagen_time/sample_imagentime_checkpoint.py \
  --config configs/ImagenTime/VerbalTS_synthetic_u.yaml \
  --model-ckpt /playpen-shared/haochenz/ImagenFew/logs/ImagenTime/VerbalTS_synthetic_u/a8bbb05c-3e69-4a54-8844-ed6ba3b00b66/ImagenTime.pt \
  --split test \
  --batch-size 256 \
  --num-variants 1 \
  --output /playpen-shared/haochenz/ImagenFew/logs/ImagenTime/VerbalTS_synthetic_u/a8bbb05c-3e69-4a54-8844-ed6ba3b00b66/sample_test.pt \
  --save-metadata
