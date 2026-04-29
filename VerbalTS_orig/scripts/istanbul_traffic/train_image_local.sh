python run_image.py \
  --cond_modal text \
  --training_stage finetune \
  --save_folder ./logs/istanbul_traffic_image_augmented/text2ts_segment \
  --model_diff_config_path configs/istanbul_traffic/diff/model_text2ts_dep.yaml \
  --model_cond_config_path configs/istanbul_traffic/cond/text_msmdiffmv.yaml \
  --train_config_path configs/istanbul_traffic/train.yaml \
  --evaluate_config_path configs/istanbul_traffic/evaluate.yaml \
  --data_folder ../data/VerbalTSDatasets/istanbul_traffic \
  --img_size 14 \
  --patch_size 2 \
  --epochs 1000 \
  --batch_size 512 \
  --n_runs 1 \
  --only_evaluate True


#python run_image.py \
#  --cond_modal text \
#  --training_stage finetune \
#  --save_folder ./logs/istanbul_traffic_image/text2ts_segment \
#  --model_diff_config_path configs/istanbul_traffic/diff/model_text2ts_dep.yaml \
#  --model_cond_config_path configs/istanbul_traffic/cond/text_msmdiffmv.yaml \
#  --train_config_path configs/istanbul_traffic/train.yaml \
#  --evaluate_config_path configs/istanbul_traffic/evaluate.yaml \
#  --data_folder ../data/VerbalTSDatasets/istanbul_traffic \
#  --img_size 14 \
#  --patch_size 2 \
#  --epochs 1000 \
#  --batch_size 512 \
#  --n_runs 1 \
#  --only_evaluate True