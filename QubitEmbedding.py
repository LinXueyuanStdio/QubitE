from typing import List, Tuple

import torch
from torch import nn

from toolbox.nn.ComplexAttention import ComplexLinear
from toolbox.nn.ComplexEmbedding import (
    ComplexEmbedding,
    ComplexDropout,
    ComplexBatchNorm1d,
    ComplexMult,
    ComplexAdd,
    ComplexSubstract,
    ComplexDiv,
    ComplexConjugate,
    ComplexNum
)

Qubit = Tuple[ComplexNum, ComplexNum]


class QubitEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim, num_channels, norm_num_channels=2):
        super(QubitEmbedding, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.embeddings = nn.ModuleList([ComplexEmbedding(num_entities, embedding_dim, norm_num_channels) for _ in range(num_channels)])

    def forward(self, idx):
        embedings = []
        for embedding in self.embeddings:
            embedings.append(embedding(idx))
        return tuple(embedings)

    def init(self):
        for embedding in self.embeddings:
            embedding.init()

    def get_embeddings(self):
        return [embedding.get_embeddings() for embedding in self.embeddings]

    def get_cat_embedding(self):
        return torch.cat([embedding.get_cat_embedding() for embedding in self.embeddings], 1)


class QubitDropout(nn.Module):
    def __init__(self, dropout_rate_list: List[List[float]]):
        super(QubitDropout, self).__init__()
        self.dropout_rate_list = dropout_rate_list
        self.dropouts = nn.ModuleList([ComplexDropout(dropout_rate) for dropout_rate in dropout_rate_list])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.dropouts[idx](complex_number))
        return tuple(out)


class QubitBatchNorm1d(nn.Module):
    def __init__(self, embedding_dim, num_channels, norm_num_channels=2):
        super(QubitBatchNorm1d, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.batch_norms = nn.ModuleList([ComplexBatchNorm1d(embedding_dim, norm_num_channels) for _ in range(num_channels)])

    def forward(self, complex_numbers):
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            out.append(self.batch_norms[idx](complex_number))
        return tuple(out)


class QubitNorm(nn.Module):
    def forward(self, e: Qubit) -> Qubit:
        (alpha_a, alpha_b), (beta_a, beta_b) = e
        length = torch.sqrt(alpha_a ** 2 + alpha_b ** 2 + beta_a ** 2 + beta_b ** 2)
        return (alpha_a / length, alpha_b / length), (beta_a / length, beta_b / length)


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

    def forward(self, h: Qubit, r: Qubit) -> Qubit:
        h_a, h_b = self.norm(h)
        r_a, r_b = self.norm(r)
        a = self.complex_sub(self.complex_mul(h_a, r_a), self.complex_mul(h_b, self.complex_conj(r_b)))
        b = self.complex_add(self.complex_mul(h_a, r_b), self.complex_mul(h_b, self.complex_conj(r_a)))
        return a, b


class QubitScoringAll(nn.Module):
    def forward(self, complex_numbers, embeddings):
        e_a, e_b = embeddings
        out = []
        for idx, complex_number in enumerate(list(complex_numbers)):
            t_a, t_b = complex_number
            a = torch.mm(t_a, e_a[idx].transpose(1, 0))
            b = torch.mm(t_b, e_b[idx].transpose(1, 0))
            out.append((a, b))
        return tuple(out)


class QubitProjection(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    out in C^d_out, out_a in (B, d_out), out_b in (B, d_out)
    """

    def __init__(self, in_features, out_features):
        super(QubitProjection, self).__init__()
        self.W_a = ComplexLinear(in_features, out_features)
        self.W_b = ComplexLinear(in_features, out_features)

    def forward(self, x):
        x_a, x_b = x
        out_a = self.W_a(x_a)
        out_b = self.W_b(x_b)
        return out_a, out_b


class QubitLinear(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    out in C^d_out, out_a in (B, d_out), out_b in (B, d_out)
    """

    def __init__(self, in_features, out_features):
        super(QubitLinear, self).__init__()
        self.W_a = ComplexLinear(in_features, out_features)
        self.W_b = ComplexLinear(in_features, out_features)

    def forward(self, x):
        x_a, x_b = x
        out_a = self.W_a(x_a) - self.W_b(x_b)
        out_b = self.W_a(x_b) + self.W_b(x_a)
        return out_a, out_b


class QubitRealLinear(nn.Module):
    """
    x = x_a + x_b i
    W = W_a + W_b i
    W * x = (W_a * x_a - W_b * x_b) + (W_a * x_b + W_b * x_a) i

    x in C^d, x_a in (B, d), x_b in (B, d)
    W in C^(d, d_out), W_a in (d, d_out), W_b in (d, d_out)
    out in C^d_out, out_a in (B, d_out), out_b in (B, d_out)
    """

    def __init__(self, in_features, out_features):
        super(QubitRealLinear, self).__init__()
        self.W = ComplexLinear(in_features, out_features)

    def forward(self, x):
        x_a, x_b = x
        out_a = self.W(x_a)
        out_b = self.W(x_b)
        return out_a, out_b
