cd ..
export CUDA_VISIBLE_DEVICES=0,1
PORT=10001


BackBone='ViT-B/32'
BackBoneName='ViT-B32'
LoraRank=16
LR=0.01
WD=0.5
EPOCH=10
STEP=3000
SEED=0
BS=64
dataset1='COCO_neg_pos'
tag="test"
result_path="./results/CLIP_FT_unique_imageid/${BackBoneName}"
ckpt="$result_path/${dataset1}_${tag}/CLIP_${BackBoneName}_finetune_ALL_LoraRank${LoraRank}_LR$(printf "%.8f" $LR)_WD$(printf "%.4f" $WD)_epochs${EPOCH}_steps${STEP}_seed${SEED}.ckpt"


python main.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--data_type $dataset1 \
--annotation_path /path/to/captions_train2014_unique_imageid_separate_ids_separate_captions_Add5NegCaption1_Add5PosCaption1.csv \
--data_path /path/to/COCO/2014/train2014 /path/to/COCO/2014/synthetic_negative /path/to/COCO/2014/synthetic_negative /path/to/COCO/2014/synthetic_negative /path/to/COCO/2014/synthetic_negative /path/to/COCO/2014/synthetic_negative /path/to/COCO/2014/synthetic_positive /path/to/COCO/2014/synthetic_positive /path/to/COCO/2014/synthetic_positive /path/to/COCO/2014/synthetic_positive /path/to/COCO/2014/synthetic_positive \
--ori_text_cols 2 6 10 14 18 --negative_text_cols 3 7 11 15 19 --positive_text_cols 4 8 12 16 20 --random_select_cols \
--folder_name $result_path \
--base_lr $LR \
--weight_decay $WD \
--epochs $EPOCH \
--stop_steps $STEP \
--batch_size $BS \
--sigmoid \
--style_transfer adain \
--style_transfer_alpha 0.5 \
--margin_weight 0.01 \
--margin_thre 0.5 \
--margin_additional_real_weight 10.0 \
--margin_low -2.0 \
--margin_scale 4.0 \
--tag $tag \
--master_port $PORT \
--seed $SEED



echo "########## CIFAR10 ##########"

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset CIFAR10 \
--dataset_path /path/to/cifar-10 \
--batch_size $BS \
--num_workers 4 \
--reuse_caption


echo "########## ARO Val ##########"

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VG_Relation_Val \
--dataset_path /path/to/ARO \
--batch_size $BS \
--num_workers 4

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VG_Attribution_Val \
--dataset_path /path/to/ARO \
--batch_size $BS \
--num_workers 4


echo "########## ARO ##########"

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VG_Relation \
--dataset_path /path/to/ARO \
--batch_size $BS \
--num_workers 4

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VG_Attribution \
--dataset_path /path/to/ARO \
--batch_size $BS \
--num_workers 4


echo "########## VL-CheckList ##########"

for subset in {'action','color','material','size','state'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VLCheckList \
--dataset_path /path/to/VL-CheckList \
--annotation_path /home/haoxin003/work/dataset_config/VL-CheckList/annotations/corpus/Attribute/$subset \
--batch_size $BS \
--num_workers 4

done


for subset in {'action','spatial'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VLCheckList \
--dataset_path /path/to/VL-CheckList \
--annotation_path /home/haoxin003/work/dataset_config/VL-CheckList/annotations/corpus/Relation/$subset \
--batch_size $BS \
--num_workers 4

done


for subset in {'center','margin','mid'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VLCheckList \
--dataset_path /path/to/VL-CheckList \
--annotation_path /home/haoxin003/work/dataset_config/VL-CheckList/annotations/corpus/Object/Location/$subset \
--batch_size $BS \
--num_workers 4

done


for subset in {'large','medium','small'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset VLCheckList \
--dataset_path /path/to/VL-CheckList \
--annotation_path /home/haoxin003/work/dataset_config/VL-CheckList/annotations/corpus/Object/Size/$subset \
--batch_size $BS \
--num_workers 4

done


echo "########## SugarCrepe ##########"

for subset in {'add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset SugarCrepe \
--dataset_path /path/to/COCO/2014/val2014 \
--annotation_path /path/to/SugarCrepe/$subset.json \
--batch_size $BS \
--num_workers 4

done


echo "########## SugarCrepe_pp ##########"

for subset in {'data_replace_att','data_replace_obj','data_replace_rel','data_swap_att','data_swap_obj'}
do

python eval.py \
--backbone_name $BackBone \
--lora_r $LoraRank \
--checkpoint $ckpt \
--sigmoid \
--dataset SugarCrepe_pp \
--dataset_path /path/to/COCO/2014/val2014 \
--annotation_path /path/to/SugarCrepe_pp/$subset.json \
--batch_size $BS \
--num_workers 4

done
