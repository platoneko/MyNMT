#!/usr/bin/env bash
python ./ranker_train.py --train_path=./dataset/opensub_2004k.json \
--valid_path=./dataset/bigbang_valid.json \
--vocab_dir=./vocab \
--pretrained_vector=./pretrained.pt \
--embedding_size=500 \
--margin=1.0 \
--optimizer=Adam \
--lr=0.001 \
--num_epochs=20 \
--lr_decay=0.5 \
--valid_metric=-loss \
--gpu=0 \
--log_steps=100 \
--valid_steps=3000 \
--batch_size=32 \
--save_dir=./outputs/ranker_2004