#!/usr/bin/env bash
python ./train.py --train_path=./dataset/subtitle_train_debug.json \
--valid_path=./dataset/subtitle_train_debug.json \
--gpu=0 \
--batch_size=8 \
--min_freq=5 \
--max_vocab_size=50000 \
--log_steps=100 \
--valid_step=200 \
--model=Seq2Seq \
--vocab_dir=./vocab/pretrain/ \
--vector_dir=./vector/pretrain/ \
--save_dir=./outputs/pretrain/ \
--dropout=0.2 \
--teach=0.5


# --valid_metric=-nll_2 \
# --ckpt=./outputs/best