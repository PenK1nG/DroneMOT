
cd ..
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=5 \
--use_env train_dis.py \
--exp_id 'test' \
--arch 'dronemot_34' \
--batch_size 1 \
--gpus '0' \
--num_epochs 30 \
--reid_cls_ids '0,1,2,3,4,5,6,7,8,9'


# ----------------------1~10 object classes are what we need
# pedestrian      (1),  --> 0
# people          (2),  --> 1
# bicycle         (3),  --> 2
# car             (4),  --> 3
# van             (5),  --> 4
# truck           (6),  --> 5
# tricycle        (7),  --> 6
# awning-tricycle (8),  --> 7
# bus             (9),  --> 8
# motor           (10), --> 9
# ----------------------