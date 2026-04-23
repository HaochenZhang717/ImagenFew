DATASET=ettm1 \
SPLIT=train \
MAX_SAMPLES=20 \
CHECKPOINT=/playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage1_qwen25_3b/joint_caption_best.pt \
bash generate_caption_train_compare.sh


#DATASET=ettm1 \
#SPLIT=train \
#MAX_SAMPLES=20 \
#CHECKPOINT=/playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage1_qwen25_3b/joint_caption_latest.pt \
#bash generate_caption_train_compare.sh



#DATASET=ettm1 \
#SPLIT=train \
#MAX_SAMPLES=20 \
#CHECKPOINT=/playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage1_qwen25_3b_ve/joint_caption_best.pt \
#bash generate_caption_train_compare.sh


