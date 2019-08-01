from torchtext.data import Field, NestedField
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

fields = {
    'post': ('post', post_field),
    'response': ('response', response_field),
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

data_iter = BucketIterator(
    data,
    batch_size=10,
    device=-1
)

post_tokens = 0
response_tokens = 0
post_unk_count = 0
response_unk_count = 0
for batch in data_iter:
    post_unk_count += (batch.post == 0).sum().item()
    response_unk_count += (batch.response == 0).sum().item()
    post_tokens += (batch.post != 1).sum().item()
    response_tokens += (batch.response != 1).sum().item()
print(post_unk_count, response_unk_count)
print(post_tokens, response_tokens)
