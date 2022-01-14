import torch
import torch.nn.functional as F
from torch import nn


class CoreTransE(nn.Module):
    def __init__(self):
        super(CoreTransE, self).__init__()

    def forward(self, h, r):
        x = h + r
        x = F.relu(x)
        return x


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, hidden_dropout=0.2):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(num_entities, embedding_dim)
        self.R = nn.Embedding(num_relations, embedding_dim)

        self.w1 = nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        self.w2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout)
        self.b = nn.Parameter(torch.zeros(num_entities))
        self.m = nn.PReLU()

        self.loss = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, h_idx, r_idx):
        h = self.E(h_idx)  # Bxd
        r = self.R(r_idx)  # Bxd

        t = self.w2(F.relu(self.w1(torch.cat([h, r], dim=-1))))
        t = t.view(-1, self.embedding_dim)

        x = torch.mm(t, self.dropout(self.E.weight).transpose(1, 0))
        x = x + self.b.expand_as(x)
        x = torch.sigmoid(x)
        return x  # batch_size x E
