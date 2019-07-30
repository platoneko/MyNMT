#!/usr/bin/env bash
python ./train.py --train_path=./dataset/opensub_valid.tsv \
--valid_path=./dataset/opensub_valid.tsv \
--gpu=0 \
--batch_size=10 \
--log_steps=100 \
--valid_step=1000 \
--min_freq=1 \
--max_vocab_size=50000 \
--model=Seq2Seq \
--save_dir=./outputs/antianti/ \
--vocab_dir=./vocab \
--dropout=0.2 \
--teach=1.0 \
--lr_decay=0.5 \
--optimizer=Adam \
--lr=0.001 \
--valid_metric=-ppx \
--mmi_anti=True \
--anti_gamma=2 \
--anti_rate=0.5

# --ckpt=./outputs/best

# --valid_metric=-nll_2 \
# --ckpt=./outputs/best