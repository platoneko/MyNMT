#!/usr/bin/env bash
python ./test.py --data_path=./dataset/opensub_valid.tsv \
--gpu=-1 \
--batch_size=32 \
--beam_size=1 \
--model=Baidu \
--vocab_dir=./vocab \
--ckpt=./outputs/best.model