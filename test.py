import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torchtext.data import Field, NestedField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

from models.seq2seq import Seq2Seq
from utils.generator import Generator

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
    data_arg.add_argument("--data_path", type=str, required=True)
    data_arg.add_argument("--vector_dir", type=str, default="./dataset/vector")
    data_arg.add_argument("--vocab_dir", type=str, default="./dataset/vocab")

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument("--embedding_size", "--embed_size", type=int, default=300)
    model_arg.add_argument("--hidden_size", type=int, default=800)
    model_arg.add_argument("--num_layers", type=int, default=2)
    model_arg.add_argument("--dropout", type=float, default=0.2)
    model_arg.add_argument("--teaching_force_rate", "--teach", type=float, default=0.5)

    # Generator
    generator_arg = parser.add_argument_group("Generator")
    generator_arg.add_argument("--beam_size", type=int, default=4)
    generator_arg.add_argument("--per_node_beam_size", type=int)


    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=-1)
    misc_arg.add_argument("--batch_size", type=int, default=32)
    misc_arg.add_argument("--ckpt", type=str, default="./outputs/best.model")
    data_arg.add_argument("--save_dir", type=str, default="./results")

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

    test_data = TabularDataset(
        path=config.data_path,
        format='json',
        fields=fields
    )

    with open(os.path.join(config.vocab_dir, 'vocab.pkl'), 'rb') as vocab_file:
        text_field.vocab = pickle.load(vocab_file)
        tag_field.vocab = text_field.vocab
    with open(os.path.join(config.vocab_dir, 'loc.pkl'), 'rb') as loc_file:
        loc_field.vocab = pickle.load(loc_file)
    with open(os.path.join(config.vocab_dir, 'gender.pkl'), 'rb') as gender_file:
        gender_field.vocab = pickle.load(gender_file)

    test_iter = BucketIterator(
        test_data,
        batch_size=config.batch_size,
        device=device,
        shuffle=True
    )

    # Model definition
    embedding = nn.Embedding(len(text_field.vocab), config.embedding_size)
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
    model.load(filename=config.ckpt)
    model.to(device)

    # Save directory
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    # Logger definition
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    fh = logging.FileHandler(os.path.join(config.save_dir, "test.log"))
    logger.addHandler(fh)

    # Generator definition
    if config.per_node_beam_size is None:
        config.per_node_beam_size = config.beam_size
    generator = Generator(
        model=model,
        end_index=model.end_index,
        padding_index=model.padding_index,
        data_iter=test_iter,
        vocab=text_field.vocab,
        logger=logger,
        beam_size=config.beam_size,
        per_node_beam_size=config.per_node_beam_size,
        result_path=os.path.join(config.save_dir, "result.txt")
    )

    # Save config
    params_file = os.path.join(config.save_dir, "params.json")
    with open(params_file, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
    print("Saved params to '{}'".format(params_file))
    logger.info(model)

    generator.generate()
    logger.info("Testing done!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program earlier!")