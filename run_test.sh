#!/usr/bin/env bash
python ./test.py --data_path=./dataset/test.json \
--gpu=0 \
--batch_size=32 \
--beam_size=1 \
--model=rnn \
--vocab_dir=./vocab \
--ckpt=./outputs/best.model \
--save_dir=./results/