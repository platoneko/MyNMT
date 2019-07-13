import torch
import torch.nn as nn

device = torch.device(0)
embedding = nn.Embedding(10, 3)
embedding.weight = nn.Parameter(torch.ones(10, 3))
embedding.to(device)

print(embedding.weight.data.device)
