"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/14
@description: null
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from QubitEmbedding import QubitBatchNorm1d, QubitDropout, QubitScoringAll, QubitNorm, QubitMult
from toolbox.nn.CoPER import ContextualParameterGenerator
from toolbox.nn.ComplexEmbedding import ComplexAlign
from toolbox.nn.Regularizer import N3


class ComplexEmbeddingGenerator(nn.Module):

    def __init__(self, generate_dim, embedding_dim, num_channels=2):
        super().__init__()
        self.num_channels = num_channels
        self.generators = nn.ModuleList([
            ContextualParameterGenerator(generate_dim, [embedding_dim])
            for _ in range(num_channels)
        ])

    def forward(self, h):
        # h: Bxd
        embedding = (f(h) for f in self.generators)
        return embedding


class QubitEmbeddingGenerator(nn.Module):

    def __init__(self, generate_dim, embedding_dim, num_channels=2):
        super().__init__()
        self.num_channels = num_channels
        self.generators = nn.ModuleList([
            ComplexEmbeddingGenerator(generate_dim, embedding_dim)
            for _ in range(num_channels)
        ])

    def forward(self, h):
        # h: Bxd
        embedding = (f(h) for f in self.generators)
        return embedding


class GeneratedQubitEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, num_channels=2):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = nn.Embedding(num_embeddings, embedding_dim // 2)  # alpha = a + bi, beta = c + di
        self.generator = QubitEmbeddingGenerator(embedding_dim // 2, embedding_dim)

    def forward(self, h_idx):
        # h: Bx1
        h = self.embedding(h_idx)
        embedding = self.generator(h)
        (ha, hai), (hb, hbi) = embedding
        # norm = ha ** 2 + hai ** 2 + hb ** 2 + hbi ** 2
        # ha /= norm
        # hai /= norm
        # hb /= norm
        # hbi /= norm
        return (ha, hai), (hb, hbi)

    def init(self):
        nn.init.xavier_normal_(self.embedding.weight.data)

    def get_embeddings(self):
        h = self.embedding.weight
        embedding = self.generator(h)
        (ha, hai), (hb, hbi) = embedding
        # norm = ha ** 2 + hai ** 2 + hb ** 2 + hbi ** 2
        # ha /= norm
        # hai /= norm
        # hb /= norm
        # hbi /= norm
        return [(ha, hai), (hb, hbi)]


class QubitE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3, regularization_weight=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.bce = nn.BCELoss()
        self.E = GeneratedQubitEmbedding(self.num_entities, self.embedding_dim)  # alpha = a + bi, beta = c + di
        self.R = GeneratedQubitEmbedding(self.num_relations, self.embedding_dim)  # alpha = a + bi, beta = c + di
        self.E_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.R_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.hidden_dp = QubitDropout([[hidden_dropout, hidden_dropout]] * 2)
        self.E_bn = QubitBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = QubitBatchNorm1d(self.embedding_dim, 4)
        self.b_x = nn.Parameter(torch.zeros(num_entities))
        self.b_y = nn.Parameter(torch.zeros(num_entities))
        # self.proj_t = QubitProjection(self.embedding_dim, self.embedding_dim)
        self.norm = QubitNorm()

        self.mul = QubitMult(norm_flag)
        # self.mul = QubitMatrixMult(norm_flag)
        # self.mul = QubitUnitaryMult(norm_flag)
        self.scoring_all = QubitScoringAll()
        self.align = ComplexAlign()
        self.regularizer = N3(regularization_weight)

    def forward(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        return self.forward_head_batch(h_idx, r_idx)

    def forward_head_batch(self, h_idx, r_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x) | x in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(h_idx)
        r = self.R(r_idx)
        # h = self.norm(h)
        h = self.E_bn(h)
        # r = self.norm(r)
        # r = self.R_bn(r)
        t = self.mul(h, r)
        # t = self.proj_t(t)

        E = self.E.get_embeddings()
        # E = self.norm(E)
        # E = self.proj_h(E)
        E = self.E_bn(E)

        score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(E))
        score_a_a, score_a_b = score_a
        y_a = score_a_a + score_a_b
        y_a = y_a + self.b_x.expand_as(y_a)

        score_b_a, score_b_b = score_b
        y_b = score_b_a + score_b_b
        y_b = y_b + self.b_y.expand_as(y_b)

        y_a = torch.sigmoid(y_a)
        y_b = torch.sigmoid(y_b)

        return y_a, y_b

    def loss(self, target, y):
        y_a, y_b = target
        return self.bce(y_a, y) + self.bce(y_b, y)

    def regular_loss(self, h_idx, r_idx, t_idx):
        h = self.E(h_idx)
        r = self.R(r_idx)
        t = self.E(t_idx)
        (h_a, h_a_i), (h_b, h_b_i) = h
        (r_a, r_a_i), (r_b, r_b_i) = r
        (t_a, t_a_i), (t_b, t_b_i) = t
        factors = (
            torch.sqrt(h_a ** 2 + h_a_i ** 2 + h_b ** 2 + h_b_i ** 2),
            torch.sqrt(r_a ** 2 + r_a_i ** 2 + r_b ** 2 + r_b_i ** 2),
            torch.sqrt(t_a ** 2 + t_a_i ** 2 + t_b ** 2 + t_b_i ** 2),
        )
        regular_loss = self.regularizer(factors)
        return regular_loss

    def reverse_loss(self, h_idx, r_idx, max_relation_idx):
        h = self.E(h_idx)
        h_a, h_b = h
        h = (h_a.detach(), h_b.detach())

        r = self.R(r_idx)
        reverse_rel_idx = (r_idx + max_relation_idx) % (2 * max_relation_idx)

        t = self.mul(h, r)
        reverse_r = self.R(reverse_rel_idx)
        reverse_t = self.mul(t, reverse_r)
        reverse_a, reverse_b = self.align(reverse_t, h)  # a + b i
        reverse_score = reverse_a + reverse_b
        reverse_score = torch.mean(F.relu(reverse_score))

        return reverse_score

    def init(self):
        self.E.init()
        self.R.init()


if __name__ == "__main__":
    B = 5
    E = 10
    R = 10
    import random

    h = torch.LongTensor(random.choices([[i] for i in range(E)], k=B))
    r = torch.LongTensor(random.choices([[i] for i in range(R)], k=B))
    t = torch.LongTensor(random.choices([[i] for i in range(E)], k=B))
    target = torch.rand((B, E))
    model = QubitE(E, R, 6)
    model.init()
    pred = model(h, r)
    print(pred)
    print(model.loss(pred, target))
    print(model.regular_loss(h, r, t))
