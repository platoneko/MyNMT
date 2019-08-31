#!/usr/bin/env bash
python ./test.py --data_path=./dataset/opensub_valid.json \
--gpu=0 \
--batch_size=32 \
--beam_size=1 \
--model=Standard \
--vocab_dir=./vocab \
--ckpt=./outputs/normal/best.model \
--save_dir=./results/normal/