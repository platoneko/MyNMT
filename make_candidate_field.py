from models.ranker import EmbeddingRanker, RNNRanker
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import pickle
import torch
import torch.nn as nn


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# args
DATA_PATH = "./dataset/destylized_bigbang.json"
CANDIDATE_LIB = "./candidate_lib.pt"
PRETRAINED_RANKER = "./outputs/rnn_ranker/best.model"
EMBEDDING_SIZE = 500
MARGIN = 1.0
TOPK = 20
GPU = 0
BATCH_SIZE = 32


if __name__ == "__main__":
    if torch.cuda.is_available() and GPU >= 0:
        device = torch.device(GPU)
    else:
        device = torch.device('cpu')

    tokenizer = lambda x: x.split()

    post_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        include_lengths=True
    )

    fields = {
        'post': ('post', post_field),
    }

    data = TabularDataset(
        path=DATA_PATH,
        format='json',
        fields=fields
    )

    with open('./vocab/post.vocab.pkl', 'rb') as post_vocab:
        post_field.vocab = pickle.load(post_vocab)
    with open('./vocab/response.vocab.pkl', 'rb') as response_vocab:
        response_vocab = pickle.load(response_vocab)

    data_iter = BucketIterator(
        data,
        batch_size=BATCH_SIZE,
        device=device,
        shuffle=False
    )

    post_embedding = nn.Embedding(len(post_field.vocab), EMBEDDING_SIZE)
    response_embedding = nn.Embedding(len(response_vocab), EMBEDDING_SIZE)
    ranker = RNNRanker(
        EMBEDDING_SIZE,
        post_embedding,
        response_embedding,
        padding_idx=response_vocab.stoi[PAD_TOKEN],
        margin=MARGIN)
    ranker.load(PRETRAINED_RANKER)
    ranker.to(device)
    ranker.eval()

    candidate_lib = torch.load(CANDIDATE_LIB)
    assert isinstance(candidate_lib, tuple)
    candidate_lib = tuple(item.to(device) for item in candidate_lib)

    _num_candidates = candidate_lib[0].size(0)
    _start = 0
    _end = min(_start + BATCH_SIZE, _num_candidates)
    _num_batches = _num_candidates // BATCH_SIZE + 1
    candidate_vector_list = []
    with torch.no_grad():
        for _ in range(_num_batches):
            candidate_vector = ranker.get_candidate_vector((candidate_lib[0][_start:_end], candidate_lib[1][_start:_end]))
            candidate_vector_list.append(candidate_vector)
            _start = _end
            _end = min(_start + BATCH_SIZE, _num_candidates)
    candidate_vector_lib = torch.cat(candidate_vector_list)

    candidate_list = []
    for batch in data_iter:
        batch_score, batch_indices = ranker.rank_with_tensor(batch.post, candidate_vector_lib, TOPK)
        for indices in batch_indices:
            candidate_list.append(indices.tolist())

    with open("candidate_field.list.pkl", 'wb') as tgt:
        pickle.dump(candidate_list, tgt)
