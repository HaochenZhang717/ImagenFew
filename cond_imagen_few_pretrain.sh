CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project CondImagenFewPretrain \
--config ./configs/conditional_pretrain/pretrain.yaml \
--no_test_model