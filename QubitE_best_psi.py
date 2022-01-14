"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/14
@description: null
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from QubitEmbedding import QubitBatchNorm1d, QubitDropout, QubitEmbedding, QubitScoringAll, QubitNorm, BatchQubitScoringAll, Qubit
from toolbox.nn.ComplexEmbedding import ComplexAlign, ComplexMult, ComplexAdd, ComplexSubstract, ComplexDiv, ComplexConjugate
from toolbox.nn.Regularizer import N3


class QubitMult(nn.Module):
    """
    U[r] = [[r_a, -^r_b],
            [r_b, ^r_a ]]  ^ is conj
    h = [h_a, h_b]

    h_a, h_b in CP^d
    r_a, r_b in CP^d

    h * r = U[r] * h = [r_a * h_a + -^r_b * h_b, r_b * h_a + ^r_a * h_b]
    """

    def __init__(self, norm_flag=False):
        super(QubitMult, self).__init__()
        self.norm_flag = norm_flag
        self.complex_mul = ComplexMult(False)
        self.complex_add = ComplexAdd()
        self.complex_sub = ComplexSubstract()
        self.complex_div = ComplexDiv()
        self.complex_conj = ComplexConjugate()
        self.norm = QubitNorm()

    def forward(self, h: Qubit, r: Qubit, psi) -> Qubit:
        if self.norm_flag:
            h_a, h_b = self.norm(h)
            r_a, r_b = self.norm(r)
        else:
            h_a, h_b = h
            r_a, r_b = r
        z = (torch.cos(psi), torch.sin(psi))
        a = self.complex_sub(self.complex_mul(h_a, r_a), self.complex_mul(z, self.complex_mul(h_b, self.complex_conj(r_b))))
        b = self.complex_add(self.complex_mul(h_a, r_b), self.complex_mul(z, self.complex_mul(h_b, self.complex_conj(r_a))))
        return a, b

class QubitE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.1, hidden_dropout=0.1, regularization_weight=0.1):
        super(QubitE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.bce = nn.BCELoss()
        self.E = QubitEmbedding(self.num_entities, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.R = QubitEmbedding(self.num_relations, self.embedding_dim, 2)  # alpha = a + bi, beta = c + di
        self.R_psi = nn.Embedding(self.num_relations, self.embedding_dim)
        self.E_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.R_dropout = QubitDropout([[input_dropout, input_dropout]] * 2)
        self.hidden_dp = QubitDropout([[hidden_dropout, hidden_dropout]] * 2)
        self.E_bn = QubitBatchNorm1d(self.embedding_dim, 2)
        self.R_bn = QubitBatchNorm1d(self.embedding_dim, 4)
        self.b_a = nn.Parameter(torch.zeros(num_entities))
        self.b_x = nn.Parameter(torch.zeros(num_entities))
        self.b_y = nn.Parameter(torch.zeros(num_entities))
        self.b_z = nn.Parameter(torch.zeros(num_entities))
        # self.proj_t = QubitProjection(self.embedding_dim, self.embedding_dim)
        self.norm = QubitNorm()

        self.mul = QubitMult(norm_flag)
        # self.mul = QubitMatrixMult(norm_flag)
        # self.mul = QubitUnitaryMult(norm_flag)
        self.scoring_all = QubitScoringAll()
        self.batch_scoring_all = BatchQubitScoringAll()
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
        h = self.norm(h)
        h = self.E_bn(h)

        r = self.R(r_idx)
        r = self.norm(r)
        r_psi = self.R_psi(r_idx)

        t = self.mul(h, r, r_psi)

        E = self.E.get_embeddings()
        E = self.norm(E)
        E = self.E_bn(E)

        score_a, score_b = self.scoring_all(self.E_dropout(t), self.E_dropout(E))
        score_a_a, score_a_b = score_a
        score_b_a, score_b_b = score_b

        y_a = torch.sigmoid(score_a_a + self.b_a.expand_as(score_a_a))
        y_ai = torch.sigmoid(score_a_b + self.b_x.expand_as(score_a_b))
        y_b = torch.sigmoid(score_b_a + self.b_y.expand_as(score_b_a))
        y_bi = torch.sigmoid(score_b_b + self.b_z.expand_as(score_b_b))

        return y_a, y_ai, y_b, y_bi

    def loss(self, target, y):
        y_a, y_ai, y_b, y_bi = target
        return self.bce(y_a, y) + self.bce(y_ai, y) + self.bce(y_b, y) + self.bce(y_bi, y)

    def forward_tail_batch(self, r_idx, t_idx):
        r_idx = r_idx.view(-1)
        t_idx = t_idx.view(-1)
        t = self.E(t_idx)
        t = self.norm(t)
        t = self.E_bn(t)
        (t1, t2), (t3, t4) = t
        t1 = t1.unsqueeze(dim=-1)
        t2 = t2.unsqueeze(dim=-1)
        t3 = t3.unsqueeze(dim=-1)
        t4 = t4.unsqueeze(dim=-1)  # B, d, 1
        t = (t1, t2), (t3, t4)

        r = self.R(r_idx)
        r = self.norm(r)
        (r1, r2), (r3, r4) = r
        r1 = r1.unsqueeze(dim=1)
        r2 = r2.unsqueeze(dim=1)
        r3 = r3.unsqueeze(dim=1)
        r4 = r4.unsqueeze(dim=1)  # B, 1, d
        r = (r1, r2), (r3, r4)
        r_psi = self.R_psi(r_idx)
        r_psi = r_psi.unsqueeze(dim=1)

        E = self.E.get_embeddings()
        E = self.norm(E)
        E = self.E_bn(E)
        (E1, E2), (E3, E4) = E
        E1 = E1.unsqueeze(dim=0)
        E2 = E2.unsqueeze(dim=0)
        E3 = E3.unsqueeze(dim=0)
        E4 = E4.unsqueeze(dim=0)  # 1, N, d
        E = (E1, E2), (E3, E4)
        h = self.mul(E, r, r_psi)  # B, N, d

        score_a, score_b = self.batch_scoring_all(self.E_dropout(t), self.E_dropout(h)) # B, N
        s1, s2 = score_a
        s3, s4 = score_b

        y_a = torch.sigmoid(s1 + self.b_a.expand_as(s1))
        y_ai = torch.sigmoid(s2 + self.b_x.expand_as(s2))
        y_b = torch.sigmoid(s3 + self.b_y.expand_as(s3))
        y_bi = torch.sigmoid(s4 + self.b_z.expand_as(s4))

        return y_a, y_ai, y_b, y_bi

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
    model = QubitE(E, R, 5)
    pred = model(h, r)
    print(pred)
    print(model.loss(pred, target))
    print(model.regular_loss(h, r, t))
