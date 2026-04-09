CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py \
--subset_p 1.0 \
--ddp \
--wandb \
--wandb_project SelfConditionalGeneration \
--config ./configs/pretrain/self_conditional.yaml