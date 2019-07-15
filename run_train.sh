#!/usr/bin/env bash
python ./train.py --train_path=./dataset/mini_train_data.json \
--valid_path=./dataset/mini_train_data.json \
--gpu=0 \
--batch_size=8 \
--min_freq=1 \
--log_steps=8 \
--valid_step=32 \
--model=Seq2Seq
# --ckpt=./outputs/best