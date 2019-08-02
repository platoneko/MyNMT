import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

from models.ranker import EmbeddingRanker
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
    data_arg.add_argument("--vocab_dir", type=str, default="./vocab")
    data_arg.add_argument("--pretrained_vector", type=str, default="./pretrained_vector.pt")

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument("--embedding_size", "--embed_size", type=int, default=500)
    model_arg.add_argument("--margin", type=float, default=1.0)

    # Training
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.001)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--num_epochs", type=int, default=20)
    train_arg.add_argument("--lr_decay", type=float, default=None)
    train_arg.add_argument("--valid_metric", type=str, default="-loss")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=-1)
    misc_arg.add_argument("--log_steps", type=int, default=200)
    misc_arg.add_argument("--valid_steps", type=int, default=800)
    misc_arg.add_argument("--batch_size", type=int, default=32)
    misc_arg.add_argument("--ckpt", type=str)
    data_arg.add_argument("--save_dir", type=str, default="./outputs/ranker")

    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = get_config()

    if torch.cuda.is_available() and config.gpu >= 0:
        device = torch.device(config.gpu)
    else:
        device = torch.device('cpu')

    # Data definition
    tokenizer = lambda x: x.split()

    post_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
    )
    response_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
    )

    fields = {
        'post': ('post', post_field),
        'response': ('response', response_field),
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

    with open(os.path.join(config.vocab_dir, 'post_vocab.pkl'), 'rb') as vocab_file:
        post_field.vocab = pickle.load(vocab_file)
    with open(os.path.join(config.vocab_dir, 'response_vocab.pkl'), 'rb') as vocab_file:
        response_field.vocab = pickle.load(vocab_file)

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
    post_embedding = nn.Embedding(len(post_field.vocab), config.embedding_size)
    response_embedding = nn.Embedding(len(response_field.vocab), config.embedding_size)
    model = EmbeddingRanker(
        config.embedding_size,
        post_embedding,
        response_embedding,
        padding_idx=response_field.vocab.stoi[PAD_TOKEN],
        margin=config.margin)
    model.to(device)

    # Optimizer definition
    assert config.optimizer in ['SGD', 'Adam']
    if config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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
        valid_metric_name=config.valid_metric,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir,
        log_steps=config.log_steps,
        valid_steps=config.valid_steps,
        grad_clip=config.grad_clip,
        lr_scheduler=lr_scheduler,
        save_summary=False)
    if config.ckpt is not None:
        trainer.load(file_prefix=config.ckpt)
    else:
        model.load(config.pretrained_vector)
    trainer.train()
    logger.info("Training done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program earlier!")