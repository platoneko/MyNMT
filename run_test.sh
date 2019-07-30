#!/usr/bin/env bash
python ./test.py --data_path=./dataset/opensub_valid.tsv \
--gpu=0 \
--batch_size=32 \
--beam_size=1 \
--model=Seq2Seq \
--vocab_dir=./vocab \
--ckpt=./outputs/antianti/best.model \
--save_dir=./results/antianti/