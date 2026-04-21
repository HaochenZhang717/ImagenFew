python scripts_eval/evaluate_saved_eval_samples.py \
  --eval-samples-dir /playpen-shared/haochenz/ImagenFew/logs/ImagenTime/VerbalTS_istanbul_traffic/8cb20851-0b7d-4388-8f42-a67b10cd1fc3/eval_samples \
  --eval-metrics disc vaeFID \
  --fid-vae-ckpt-root /playpen-shared/haochenz/ImagenFew/fid_vae_ckpts



python scripts_eval/evaluate_saved_eval_samples.py \
  --eval-samples-dir /playpen-shared/haochenz/ImagenFew/logs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_qwen3/2d9c363b-2b4d-403a-8ccd-439d581cb0ff/eval_samples \
  --eval-metrics disc vaeFID \
  --fid-vae-ckpt-root /playpen-shared/haochenz/ImagenFew/fid_vae_ckpts


