"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/14
@description: null
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from QubitEmbedding import QubitBatchNorm1d, QubitDropout, QubitEmbedding, QubitScoringAll, QubitNorm, QubitMult
from toolbox.nn.ComplexEmbedding import ComplexAlign
from toolbox.nn.Regularizer import N3


class QubitE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.2, hidden_dropout=0.3, regularization_weight=0.1):
        super(QubitE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.bce = nn.BCELoss()
        self.E = QubitEmbedding(self.num_entities, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.R = QubitEmbedding(self.num_relations, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.E_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.R_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.hidden_dp = QubitDropout([[hidden_dropout, hidden_dropout]] * 2)
        self.E_bn = QubitBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = QubitBatchNorm1d(self.embedding_dim, 2)
        self.b_x = nn.Parameter(torch.zeros(num_entities))
        self.b_y = nn.Parameter(torch.zeros(num_entities))
        self.norm = QubitNorm()

        self.mul = QubitMult(norm_flag)
        self.scoring_all = QubitScoringAll()
        self.align = ComplexAlign()
        self.regularizer = N3(regularization_weight)

    def forward(self, h_idx, r_idx):
        h_idx = h_idx.view(-1)
        r_idx = r_idx.view(-1)
        return self.forward_head_batch(h_idx, r_idx)

    def forward_head_batch(self, e1_idx, rel_idx):
        """
        Completed.
        Given a head entity and a relation (h,r), we compute scores for all possible triples,i.e.,
        [score(h,r,x) | x in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        h = self.E(e1_idx)
        r = self.R(rel_idx)
        h = self.norm(h)
        t = self.mul(h, r)
        t = self.norm(t)

        score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(self.E_bn(self.norm(self.E.get_embeddings()))))
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

    def regular_loss(self, e1_idx, rel_idx):
        h = self.E(e1_idx)
        r = self.R(rel_idx)
        (h_a, h_a_i), (h_b, h_b_i) = h
        (r_a, r_a_i), (r_b, r_b_i) = r
        factors = (
            torch.sqrt(h_a ** 2 + h_a_i ** 2 + h_b ** 2 + h_b_i ** 2),
            torch.sqrt(r_a ** 2 + r_a_i ** 2 + r_b ** 2 + r_b_i ** 2),
        )
        regular_loss = self.regularizer(factors)
        return regular_loss

    def reverse_loss(self, e1_idx, rel_idx, max_relation_idx):
        h = self.E(e1_idx)
        h_a, h_b = h
        h = (h_a.detach(), h_b.detach())

        r = self.R(rel_idx)
        reverse_rel_idx = (rel_idx + max_relation_idx) % (2 * max_relation_idx)

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
    h = torch.LongTensor([[i] for i in range(5)])
    r = torch.LongTensor([[i] for i in range(5)])
    model = QubitE(10, 10, 5)
    pred = model(h, r)
    print(pred)
    print(pred.shape)
    y = torch.rand_like(pred)
    print(model.loss(pred, y))
