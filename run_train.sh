#!/usr/bin/env bash
python ./train.py --train_path=./dataset/opensub_valid.json \
--valid_path=./dataset/opensub_valid.json \
--gpu=0 \
--batch_size=4 \
--log_steps=100 \
--valid_step=800 \
--min_freq=5 \
--max_vocab_size=50000 \
--model=Transformer \
--save_dir=./outputs/transformer \
--vocab_dir=./vocab \
--dropout=0.2 \
--lr_decay=0.5 \
--optimizer=Adam \
--lr=0.001 \
--valid_metric=-ppx \
--hidden_size=300 \
--embedding_size=300 \
--num_layers=1

