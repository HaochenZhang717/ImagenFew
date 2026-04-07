CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_no_sample.py \
--subset_p 1.0 \
--ddp \
--wandb \
--wandb_project SelfConditionalGeneration \
--config ./configs/pretrain/self_conditional.yaml