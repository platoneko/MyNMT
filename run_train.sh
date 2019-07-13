python ./train.py --train_path=./dataset/mini_train_data.json \
--valid_path=./dataset/mini_eval_data.json \
--gpu=0 \
--batch_size=32 \
--min_freq=1 \
# --ckpt=./outputs/best