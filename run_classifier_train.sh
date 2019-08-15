#!/usr/bin/env bash
python ./classifier_train.py --train_path=./dataset/delta_classifier_train.json \
--valid_path=./dataset/delta_classifier_valid.json \
--vocab_dir=./vocab \
--pretrained_vector=./pretrained.pt \
--embedding_size=500 \
--optimizer=Adam \
--lr=0.001 \
--num_epochs=20 \
--lr_decay=0.5 \
--valid_metric=-loss \
--gpu=0 \
--log_steps=100 \
--valid_steps=1000 \
--batch_size=16 \
--save_dir=./outputs/delta_classifier