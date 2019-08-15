import os
import json
import pickle
import torch
import torch.nn as nn
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from models.classifier import DeltaConvClassifier


BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# args
# SPEAKERS = ('Amy', 'Bernadette', 'Howard', 'Leonard', 'Penny', 'Raj', 'Sheldon')
DATA_DIR = './dataset'
EMBEDDING_SIZE = 500
KERNEL_SIZE = 3
PRETRAINED_CLASSIFIER = "./outputs/classifier/best.model"
TOPK = 20
UNK_SPEAKER = False
GPU = 0
BATCH_SIZE = 32


class TopkCollector(object):
    def __init__(self, classifier, data_iter, speaker, speaker_id, k=20):
        self.classifier = classifier
        self.data_iter = data_iter
        self.speaker = speaker
        self.speaker_id = speaker_id
        self.k = k
        self.num_samples = 0

        self.classifier.eval()

    def collect(self, data_dir='./dataset'):
        collector = {'vectors': [], 'probabilities': [], 'ids': []}
        with torch.no_grad():
            for batch in self.data_iter:
                batch_topk_vectors, batch_topk_probabilities, batch_topk_ids = self.iterate(batch)
                collector['vectors'].append(batch_topk_vectors)
                collector['probabilities'].append(batch_topk_probabilities)
                collector['ids'].append(batch_topk_ids)
        candidate_vectors = torch.cat(collector['vectors'])
        candidate_probabilities = torch.cat(collector['probabilities'])
        candidate_ids = torch.cat(collector['ids'])
        topk_probabilities, indices = candidate_probabilities.topk(self.k, dim=0)
        topk_vectors = candidate_vectors.index_select(0, indices)
        topk_ids = candidate_ids.index_select(0, indices).tolist()
        with open(os.path.join(data_dir, "destylized_{}.json".format(self.speaker)), 'r') as src:
            lines = src.readlines()
        with open(os.path.join(data_dir, "topk_{}.txt".format(self.speaker)), 'w') as tgt:
            for i, id in enumerate(topk_ids):
                example = json.loads(lines[id])
                tgt.write("{}\t{}\n".format(example['response'], topk_probabilities[i].item()))
        return topk_vectors

    def iterate(self, inputs):
        # `probabilities` of shape (batch_size, num_classes)
        outputs = self.classifier.forward(inputs.response, inputs.destylized)
        probability = outputs.probability[:, self.speaker_id]
        batch_size = probability.size(0)
        topk_probabilities, topk_indices = probability.topk(min(self.k, batch_size), dim=0)
        topk_vectors = outputs.vector.index_select(0, topk_indices)
        topk_ids = topk_indices + self.num_samples
        self.num_samples += batch_size
        return topk_vectors, topk_probabilities, topk_ids


if __name__ == "__main__":
    if torch.cuda.is_available() and GPU >= 0:
        device = torch.device(GPU)
    else:
        device = torch.device('cpu')

    # Data definition
    tokenizer = lambda x: x.split()

    response_field = Field(
        sequential=True,
        tokenize=tokenizer,
        lower=True,
        batch_first=True,
        include_lengths=False
    )

    fields = {
        'response': ('response', response_field),
        'destylized': ('destylized', response_field)
    }

    with open(os.path.join('./vocab', 'response.vocab.pkl'), 'rb') as response_vocab:
        response_field.vocab = pickle.load(response_vocab)
    with open(os.path.join('./vocab', 'speaker.vocab.pkl'), 'rb') as speaker_vocab:
        speaker_dict = pickle.load(speaker_vocab).itos

    response_embedding = nn.Embedding(len(response_field.vocab), EMBEDDING_SIZE)
    classifier = DeltaConvClassifier(
        embedding_size=EMBEDDING_SIZE,
        response_embedding=response_embedding,
        kernel_size=KERNEL_SIZE,
        num_classes=len(speaker_dict)+1 if UNK_SPEAKER else len(speaker_dict),
        padding_idx=response_field.vocab.stoi[PAD_TOKEN]
    )
    classifier.load(PRETRAINED_CLASSIFIER)
    classifier.to(device)
    vector_list = []
    for i in range(len(speaker_dict)):
        speaker = speaker_dict[i]

        data = TabularDataset(
            path=os.path.join(DATA_DIR, "destylized_{}.json".format(speaker)),
            format='json',
            fields=fields
        )
        data_iter = BucketIterator(
            data,
            batch_size=BATCH_SIZE,
            device=device,
            shuffle=False
        )

        collector = TopkCollector(
            classifier=classifier,
            data_iter=data_iter,
            speaker=speaker,
            speaker_id=i,
            k=TOPK)

        # shape: (k, embedding_size)
        vectors = collector.collect(data_dir=DATA_DIR)
        vector_list.append(vectors.unsqueeze(0))

    vectors = torch.cat(vector_list)
    torch.save(vectors, './style_lib.pt')
