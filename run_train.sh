#!/usr/bin/env bash
python ./train.py --train_path=./dataset/train.json \
--valid_path=./dataset/valid.json \
--gpu=0 \
--batch_size=4 \
--log_steps=100 \
--valid_step=3000 \
--min_freq=5 \
--max_vocab_size=50000 \
--model=rnn \
--save_dir=./outputs/ \
--vocab_dir=./vocab \
--dropout=0.2 \
--lr_decay=0.5 \
--optimizer=adam \
--lr=0.001 \
--valid_metric=-ppx \
--hidden_size=512 \
--embedding_size=512 \
--num_layers=2

