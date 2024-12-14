cd ..
python3 train.py \
--exp_id 'res34_corr' \
--arch 'resnet_34' \
--batch_size 4 \
--gpus '0' \
--num_epochs 30 \
--reid_cls_ids '0,1,2,3,4,5,6,7,8,9'