#!/usr/bin/env bash
python ./train.py --train_path=./dataset/opensub_2006k.json \
--valid_path=./dataset/opensub_valid.json \
--gpu=0 \
--batch_size=10 \
--log_steps=100 \
--valid_step=3000 \
--min_freq=5 \
--max_vocab_size=50000 \
--model=Seq2Seq \
--save_dir=./outputs/normal \
--vocab_dir=./vocab \
--dropout=0.2 \
--teach=1.0 \
--lr_decay=0.5 \
--optimizer=Adam \
--lr=0.001 \
--valid_metric=-ppx \

