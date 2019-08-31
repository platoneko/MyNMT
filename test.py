import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator

from models.seq2seq import Seq2Seq
from models.speaker_seq2seq import SpeakerSeq2Seq
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

    # Model
    model_arg = parser.add_argument_group("Model")
    model_arg.add_argument("--model", type=str, default='Seq2Seq')
    model_arg.add_argument("--embedding_size", "--embed_size", type=int, default=300)
    model_arg.add_argument("--hidden_size", type=int, default=600)
    model_arg.add_argument("--num_layers", type=int, default=2)
    model_arg.add_argument("--dropout", type=float, default=0.2)

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
    data_arg.add_argument("--vocab_dir", type=str, default="./dataset/vocab")

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
        include_lengths=True
    )
    response_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        init_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        include_lengths=True
    )
    speaker_field = LabelField()

    fields = {
        'post': ('post', post_field),
        'response': ('response', response_field),
        'speaker': ('speaker', speaker_field)
    }

    test_data = TabularDataset(
        path=config.data_path,
        format='json',
        fields=fields
    )

    with open(os.path.join(config.vocab_dir, 'post.vocab.pkl'), 'rb') as post_vocab:
        post_field.vocab = pickle.load(post_vocab)
    with open(os.path.join(config.vocab_dir, 'response.vocab.pkl'), 'rb') as response_vocab:
        response_field.vocab = pickle.load(response_vocab)
    with open(os.path.join(config.vocab_dir, 'speaker.vocab.pkl'), 'rb') as speaker_vocab:
        speaker_field.vocab = pickle.load(speaker_vocab)

    test_iter = BucketIterator(
        test_data,
        batch_size=config.batch_size,
        device=device,
        shuffle=False
    )

    # Model definition
    post_embedding = nn.Embedding(len(post_field.vocab), config.embedding_size)
    response_embedding = nn.Embedding(len(response_field.vocab), config.embedding_size)
    assert config.model in ['Standard']
    # assert config.model in ['Standard', 'Speaker']
    if config.model == 'Standard':
        model = Seq2Seq(
            post_embedding=post_embedding,
            response_embedding=response_embedding,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            start_index=response_field.vocab.stoi[BOS_TOKEN],
            end_index=response_field.vocab.stoi[EOS_TOKEN],
            padding_index=response_field.vocab.stoi[PAD_TOKEN],
            dropout=config.dropout,
            teaching_force_rate=0.0,
            num_layers=config.num_layers,
        )

    elif config.model == 'Speaker':
        speaker_embedding = nn.Embedding(len(speaker_field.vocab), config.embedding_size)
        model = SpeakerSeq2Seq(
            post_embedding=post_embedding,
            response_embedding=response_embedding,
            speaker_embedding=speaker_embedding,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            start_index=response_field.vocab.stoi[BOS_TOKEN],
            end_index=response_field.vocab.stoi[EOS_TOKEN],
            padding_index=response_field.vocab.stoi[PAD_TOKEN],
            dropout=config.dropout,
            teaching_force_rate=0.0,
            num_layers=config.num_layers,
        )

    model.load(filename=config.ckpt)
    model.to(device)

    # Save directory
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    # Logger definition
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    fh = logging.FileHandler(os.path.join(config.save_dir, "test.log"))
    logger.addHandler(fh)

    # Generator definition
    if config.per_node_beam_size is None:
        config.per_node_beam_size = config.beam_size
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    generator = Generator(
        model=model,
        data_iter=test_iter,
        post_vocab=post_field.vocab,
        response_vocab=response_field.vocab,
        speaker_vocab=speaker_field.vocab,
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