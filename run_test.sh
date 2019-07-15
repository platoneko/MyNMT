#!/usr/bin/env bash
python ./test.py --data_path=./dataset/mini_train_data.json \
--gpu=0 \
--batch_size=16 \
--beam_size=1 \
--model=Seq2Seq \
--ckpt=./outputs/best.model