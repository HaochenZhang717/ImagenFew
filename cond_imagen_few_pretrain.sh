CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--subset_p 1.0 \
--ddp \
--wandb \
--wandb_project CondImagenFewPretrain \
--config ./configs/conditional_pretrain/pretrain.yaml \
--no_test_model