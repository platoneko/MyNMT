from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import torch
import pickle

# args
FIX_LENGTH = 20
DATA_PATH = "./dataset/bigbang.json"


if __name__ == "__main__":
    tokenizer = lambda x: x.split()

    response_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        fix_length=FIX_LENGTH
    )

    fields = {'response': ('candidate', response_field)}
    data = TabularDataset(
        path=DATA_PATH,
        format='json',
        fields=fields
    )

    with open('./vocab/response.vocab.pkl', 'rb') as response_vocab:
        response_field.vocab = pickle.load(response_vocab)

    data_iter = BucketIterator(
        data,
        batch_size=100000,
        device='cpu'
    )

    for batch in data_iter:
        print(batch.candidate.size())
        torch.save(batch.candidate, 'candidate_lib.pt')
