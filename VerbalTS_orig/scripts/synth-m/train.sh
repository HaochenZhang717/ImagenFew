export SCHEDULER=MULTISTEP
python run.py \
    --cond_modal text \
    --training_stage finetune \
    --save_folder ./logs/synth-m/text2ts_msmdiffmv \
    --model_diff_config_path configs/synth-m/diff/model_text2ts_dep.yaml \
    --model_cond_config_path configs/synth-m/cond/text_msmdiffmv.yaml \
    --train_config_path configs/synth-m/train.yaml \
    --evaluate_config_path configs/synth-m/evaluate.yaml \
    --data_folder ../data/VerbalTSDatasets/synthetic_m \
    --clip_folder "" \
    --multipatch_num 3 \
    --L_patch_len 3 \
    --base_patch 4 \
    --epochs 700 \
    --batch_size 512 \
    --clip_cache_path ""