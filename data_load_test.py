from torchtext.data import Field, LabelField
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import pickle


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
speaker_field = LabelField()

fields = {
    'post': ('post', post_field),
    'response': ('response', response_field),
    'speaker': ('speaker', speaker_field)
}

data = TabularDataset(
    path='./dataset/bigbang.json',
    format='json',
    fields=fields
)

with open('./vocab/post_vocab.pkl', 'rb') as vocab_file:
    post_field.vocab = pickle.load(vocab_file)
with open('./vocab/response_vocab.pkl', 'rb') as vocab_file:
    response_field.vocab = pickle.load(vocab_file)

speaker_field.build_vocab(data)

data_iter = BucketIterator(
    data,
    batch_size=10,
    device=-1
)

print(len(speaker_field.vocab))
with open('./vocab/speaker_vocab.pkl', 'wb') as vocab_file:
    pickle.dump(speaker_field.vocab, vocab_file)
