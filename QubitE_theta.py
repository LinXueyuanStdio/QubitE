"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2021/10/14
@description: null
"""
from QubitEmbedding import *
from toolbox.nn.ComplexEmbedding import ComplexAlign
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

    def forward(self, h: Qubit, r: Qubit, phase) -> Qubit:
        if self.norm_flag:
            h_a, h_b = self.norm(h)
            r_a, r_b = self.norm(r)
        else:
            h_a, h_b = h
            r_a, r_b = r
        a = self.complex_sub(self.complex_mul(h_a, r_a), self.complex_mul(phase, self.complex_mul(h_b, self.complex_conj(r_b))))
        b = self.complex_add(self.complex_mul(h_a, r_b), self.complex_mul(phase, self.complex_mul(h_b, self.complex_conj(r_a))))
        return a, b


class GeneratedQubitEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.theta = nn.Embedding(num_embeddings, embedding_dim)
        self.phi = nn.Embedding(num_embeddings, embedding_dim)
        self.varphi = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, h_idx):
        # h: Bx1
        theta = self.theta(h_idx)
        phi = self.phi(h_idx)
        varphi = self.varphi(h_idx)
        ha = torch.cos(theta)
        hai = torch.sin(theta) * torch.cos(phi)
        hb = torch.sin(theta) * torch.sin(phi) * torch.cos(varphi)
        hbi = torch.sin(theta) * torch.sin(phi) * torch.sin(varphi)
        return (ha, hai), (hb, hbi)

    def init(self):
        nn.init.xavier_normal_(self.theta.weight.data)
        nn.init.xavier_normal_(self.phi.weight.data)
        nn.init.xavier_normal_(self.varphi.weight.data)

    def get_embeddings(self):
        theta = self.theta.weight
        phi = self.phi.weight
        varphi = self.varphi.weight
        ha = torch.cos(theta)
        hai = torch.sin(theta) * torch.cos(phi)
        hb = torch.sin(theta) * torch.sin(phi) * torch.cos(varphi)
        hbi = torch.sin(theta) * torch.sin(phi) * torch.sin(varphi)
        return [(ha, hai), (hb, hbi)]


class GeneratedQubitGateEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.theta = nn.Embedding(num_embeddings, embedding_dim)
        self.phi = nn.Embedding(num_embeddings, embedding_dim)
        self.varphi = nn.Embedding(num_embeddings, embedding_dim)
        self.psi = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, h_idx):
        # h: Bx1
        theta = self.theta(h_idx)
        phi = self.phi(h_idx)
        varphi = self.varphi(h_idx)
        psi = self.psi(h_idx)
        ha = torch.cos(theta)
        hai = torch.sin(theta) * torch.cos(phi)
        hb = torch.sin(theta) * torch.sin(phi) * torch.cos(varphi)
        hbi = torch.sin(theta) * torch.sin(phi) * torch.sin(varphi)
        return (ha, hai), (hb, hbi), (torch.cos(psi), torch.sin(psi))

    def init(self):
        nn.init.xavier_normal_(self.theta.weight.data)
        nn.init.xavier_normal_(self.phi.weight.data)
        nn.init.xavier_normal_(self.varphi.weight.data)
        nn.init.xavier_normal_(self.psi.weight.data)

    def get_embeddings(self):
        theta = self.theta.weight
        phi = self.phi.weight
        varphi = self.varphi.weight
        psi = self.psi.weight
        ha = torch.cos(theta)
        hai = torch.sin(theta) * torch.cos(phi)
        hb = torch.sin(theta) * torch.sin(phi) * torch.cos(varphi)
        hbi = torch.sin(theta) * torch.sin(phi) * torch.sin(varphi)
        return (ha, hai), (hb, hbi), (torch.cos(psi), torch.sin(psi))


class QubitE(nn.Module):

    def __init__(self,
                 num_entities, num_relations,
                 embedding_dim,
                 norm_flag=False, input_dropout=0.1, hidden_dropout=0.1, regularization_weight=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.bce = nn.BCELoss()
        self.E = GeneratedQubitEmbedding(self.num_entities, self.embedding_dim)  # alpha = a + bi, beta = c + di
        self.R = GeneratedQubitGateEmbedding(self.num_relations, self.embedding_dim)  # alpha = a + bi, beta = c + di
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
        ra, rb, phase = self.R(r_idx)
        # h = self.norm(h)
        h = self.E_bn(h)
        # r = self.norm(r)
        # r = self.R_bn(r)
        t = self.mul(h, (ra, rb), phase)
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
