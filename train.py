import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torchtext.data import Field, NestedField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from torchtext.vocab import Vectors

from models.seq2seq import Seq2Seq
from utils.trainer import Trainer

import pickle


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def get_config():
    """
    Get config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--train_path", type=str, required=True)
    data_arg.add_argument("--valid_path", type=str, required=True)
    data_arg.add_argument("--save_dir", type=str, default="./outputs/")
    data_arg.add_argument("--vector_dir", type=str, default="./dataset/vector")
    data_arg.add_argument("--vocab_dir", type=str, default="./dataset/vocab")
    data_arg.add_argument("--max_vocab_size", type=int, default=50000)
    data_arg.add_argument("--min_freq", type=int, default=5)

    # Model
    net_arg = parser.add_argument_group("Model")
    net_arg.add_argument("--embedding_size", "--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--num_layers", type=int, default=2)
    net_arg.add_argument("--dropout", type=float, default=0.2)
    net_arg.add_argument("--teaching_force_rate", "--teach", type=float, default=0.5)

    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--num_epochs", type=int, default=20)
    train_arg.add_argument("--lr_decay", type=float, default=None)

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=-1)
    misc_arg.add_argument("--log_steps", type=int, default=1)
    misc_arg.add_argument("--valid_steps", type=int, default=1)
    misc_arg.add_argument("--batch_size", type=int, default=32)
    misc_arg.add_argument("--ckpt", type=str)
    misc_arg.add_argument("--check", action="store_true")

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = get_config()
    if config.check:
        config.save_dir = "./tmp/"
    if torch.cuda.is_available() and config.gpu >= 0:
        device = torch.device(config.gpu)
    else:
        device = torch.device('cpu')

    # Data definition
    tokenizer = lambda x: x.split()

    text_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        include_lengths=True
    )

    tag_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
    )

    tags_field = NestedField(
        nesting_field=tag_field,
        include_lengths=True,
        tokenize=tokenizer
    )

    gender_field = Field(sequential=False)
    loc_field = Field(sequential=False)

    fields = {
        'post': ('post', text_field),
        'response': ('response', text_field),
        'gender': ('gender', gender_field),
        'loc': ('loc', loc_field),
        'tag': ('tag', tags_field)
    }

    train_data = TabularDataset(
        path=config.train_path,
        format='json',
        fields=fields
    )

    valid_data = TabularDataset(
        path=config.valid_path,
        format='json',
        fields=fields
    )

    if not os.path.exists(config.vocab_dir):
        if not os.path.exists(config.vector_dir):
            os.mkdir(config.vector_dir)
        vectors = Vectors(name='./dataset/sgns.weibo.bigram-char', cache=config.vector_dir)

        text_field.build_vocab(train_data.post,
                               train_data.response,
                               train_data.tag,
                               vectors=vectors,
                               max_size=config.max_vocab_size,
                               min_freq=config.min_freq)
        tag_field.vocab = text_field.vocab

        gender_field.build_vocab(train_data)
        loc_field.build_vocab(train_data)

        os.mkdir(config.vocab_dir)
        with open(os.path.join(config.vocab_dir, 'vocab.pkl'), 'wb') as vocab_file:
            pickle.dump(text_field.vocab, vocab_file)
        with open(os.path.join(config.vocab_dir, 'loc.pkl'), 'wb') as loc_file:
            pickle.dump(loc_field.vocab, loc_file)
        with open(os.path.join(config.vocab_dir, 'gender.pkl'), 'wb') as gender_file:
            pickle.dump(gender_field.vocab, gender_file)
    else:
        with open(os.path.join(config.vocab_dir, 'vocab.pkl'), 'rb') as vocab_file:
            text_field.vocab = pickle.load(vocab_file)
            tag_field.vocab = text_field.vocab
        with open(os.path.join(config.vocab_dir, 'loc.pkl'), 'rb') as loc_file:
            loc_field.vocab = pickle.load(loc_file)
        with open(os.path.join(config.vocab_dir, 'gender.pkl'), 'rb') as gender_file:
            gender_field.vocab = pickle.load(gender_file)

    train_iter = BucketIterator(
        train_data,
        batch_size=config.batch_size,
        device=device,
        shuffle=True
    )

    valid_iter = BucketIterator(
        valid_data,
        batch_size=config.batch_size,
        device=device
    )

    # Model definition
    embedding = nn.Embedding(len(text_field.vocab), config.embedding_size)
    embedding.weight = nn.Parameter(text_field.vocab.vectors)
    model = Seq2Seq(
        embedding=embedding,
        embedding_size=config.embedding_size,
        hidden_size=config.hidden_size,
        start_index=text_field.vocab.stoi[BOS_TOKEN],
        end_index=text_field.vocab.stoi[EOS_TOKEN],
        padding_index=text_field.vocab.stoi[PAD_TOKEN],
        dropout=config.dropout,
        teaching_force_rate=config.teaching_force_rate
    )
    model.to(device)

    # Optimizer definition
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)
    # Learning rate scheduler
    if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
        lr_scheduler = \
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=config.lr_decay,
                patience=1,
                verbose=True,
                min_lr=1e-5)
    else:
        lr_scheduler = None
    # Save directory
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    # Logger definition
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
    logger.addHandler(fh)
    # Save config
    params_file = os.path.join(config.save_dir, "params.json")
    with open(params_file, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
    print("Saved params to '{}'".format(params_file))
    logger.info(model)
    # Train
    logger.info("Training starts ...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
        valid_iter=valid_iter,
        logger=logger,
        valid_metric_name="-loss",
        num_epochs=config.num_epochs,
        save_dir=config.save_dir,
        log_steps=config.log_steps,
        valid_steps=config.valid_steps,
        grad_clip=config.grad_clip,
        lr_scheduler=lr_scheduler,
        save_summary=False)
    if config.ckpt is not None:
        trainer.load(file_prefix=config.ckpt)
    trainer.train()
    logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program earlier!")