CUDA_VISIBLE_DEVICES=2,3 python run.py \
--subset_p 1.0 \
--ddp \
--wandb \
--wandb_project CondImagenFewPretrain \
--config ./configs/conditional_imagen_few/pretrain.yaml \
--no_test_model

#CUDA_VISIBLE_DEVICES=4 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project CondImagenFewPretrain \
#--config ./configs/conditional_pretrain/pretrain.yaml \
#--no_test_model