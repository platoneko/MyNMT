#!/usr/bin/env bash
python ./train.py --train_path=./dataset/opensub_valid.tsv \
--valid_path=./dataset/opensub_valid.tsv \
--gpu=0 \
--batch_size=10 \
--log_steps=100 \
--valid_step=1000 \
--min_freq=1 \
--max_vocab_size=50000 \
--model=Baidu \
--save_dir=./outputs \
--vocab_dir=./vocab \
--dropout=0.2 \
--teach=0.5 \
--lr_decay=0.5 \
--optimizer=Adam \
--lr=0.001

# --ckpt=./outputs/best

# --valid_metric=-nll_2 \
# --ckpt=./outputs/best