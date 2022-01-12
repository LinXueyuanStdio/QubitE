import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        return self.func(head, relation, tail, batch_type), (head, tail)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score, _ = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score, ent = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization:
            # Use regularization
            regularization = args.regularization * (
                    ent[0].norm(p=2) ** 2 +
                    ent[1].norm(p=2) ** 2
            ) / ent[0].shape[0]
            loss = loss + regularization
        else:
            regularization = torch.tensor([0])

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item(),
            'regularization': regularization.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        logs_rel = defaultdict(list)  # logs for every relation

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, _ = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        rel = positive_sample[i][1].item()

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()

                        log = {
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        }
                        logs.append(log)
                        logs_rel[rel].append(log)

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        metrics_rel = defaultdict(dict)
        for rel in logs_rel:
            for metric in logs_rel[rel][0].keys():
                metrics_rel[rel][metric] = sum([log[metric] for log in logs_rel[rel]]) / len(logs_rel[rel])

        return metrics, metrics_rel


class Rotate3D(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize bias to 1
        nn.init.ones_(
            tensor=self.relation_embedding[:, 3 * hidden_dim:4 * hidden_dim]
        )

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        head_i, head_j, head_k = torch.chunk(head, 3, dim=2)
        beta_1, beta_2, theta, bias = torch.chunk(rel, 4, dim=2)
        tail_i, tail_j, tail_k = torch.chunk(tail, 3, dim=2)

        bias = torch.abs(bias)

        # Make phases of relations uniformly distributed in [-pi, pi]
        beta_1 = beta_1 / (self.embedding_range.item() / self.pi)
        beta_2 = beta_2 / (self.embedding_range.item() / self.pi)
        theta = theta / (self.embedding_range.item() / self.pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Obtain representation of the rotation axis
        rel_i = torch.cos(beta_1)
        rel_j = torch.sin(beta_1) * torch.cos(beta_2)
        rel_k = torch.sin(beta_1) * torch.sin(beta_2)

        C = rel_i * head_i + rel_j * head_j + rel_k * head_k
        C = C * (1 - cos_theta)

        # Rotate the head entity
        new_head_i = head_i * cos_theta + C * rel_i + sin_theta * (rel_j * head_k - head_j * rel_k)
        new_head_j = head_j * cos_theta + C * rel_j - sin_theta * (rel_i * head_k - head_i * rel_k)
        new_head_k = head_k * cos_theta + C * rel_k + sin_theta * (rel_i * head_j - head_i * rel_j)

        score_i = new_head_i * bias - tail_i
        score_j = new_head_j * bias - tail_j
        score_k = new_head_k * bias - tail_k

        score = torch.stack([score_i, score_j, score_k], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score


class QubitE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 4))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 5))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # Initialize bias to 1
        nn.init.ones_(
            tensor=self.relation_embedding[:, 4 * hidden_dim:5 * hidden_dim]
        )

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        ha, hai, hb, hbi = torch.chunk(head, 4, dim=2)
        ra, rai, rb, rbi, bias = torch.chunk(rel, 5, dim=2)
        ta, tai, tb, tbi = torch.chunk(tail, 4, dim=2)

        bias = torch.abs(bias)
        norm = torch.sqrt(ha ** 2 + hai ** 2 + hb ** 2 + hbi ** 2).detach()
        ha = ha / norm
        hai = hai / norm
        hb = hb / norm
        hbi = hbi / norm

        norm = torch.sqrt(ra ** 2 + rai ** 2 + rb ** 2 + rbi ** 2).detach()
        ra = ra / norm
        rai = rai / norm
        rb = rb / norm
        rbi = rbi / norm

        norm = torch.sqrt(ta ** 2 + tai ** 2 + tb ** 2 + tbi ** 2).detach()
        ta = ta / norm
        tai = tai / norm
        tb = tb / norm
        tbi = tbi / norm

        ha = ra * ha - rai * hai - rb * hb - rbi * hbi
        hai = ra * hai + rai * ha - rb * hbi + rbi * hb
        hb = rb * ha - rbi * hai + ra * hb + rai * hbi
        hbi = rbi * ha + rb * hai + ra * hbi - rai * hb

        score_a = ha * bias - ta
        score_ai = hai * bias - tai
        score_b = hb * bias - tb
        score_bi = hbi * bias - tbi

        score = torch.stack([
            score_a,
            score_ai,
            score_b,
            score_bi,
        ], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score

class ComplexMult(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h * r = (h_a * r_a - h_b * r_b) + (h_a * r_b + h_b * r_a) i

    h in C^d
    r in C^d
    """

    def __init__(self, norm_flag=False):
        super(ComplexMult, self).__init__()
        self.flag_hamilton_mul_norm = norm_flag

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        if self.flag_hamilton_mul_norm:
            # Normalize the relation to eliminate the scaling effect
            r_norm = torch.sqrt(r_a ** 2 + r_b ** 2)
            r_a = r_a / r_norm
            r_b = r_b / r_norm
        t_a = h_a * r_a - h_b * r_b
        t_b = h_a * r_b + h_b * r_a
        return t_a, t_b


class ComplexAdd(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h + r = (h_a + r_a) + (h_b + r_b) i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexAdd, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        t_a = h_a + r_a
        t_b = h_b + r_b
        return t_a, t_b


class ComplexConjugate(nn.Module):
    """
    h = h_a + h_b i
    ^h = h_a - h_b i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexConjugate, self).__init__()

    def forward(self, h):
        h_a, h_b = h
        return h_a, -h_b


class ComplexSubstract(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h - r = (h_a - r_a) + (h_b - r_b) i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexSubstract, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        t_a = h_a - r_a
        t_b = h_b - r_b
        return t_a, t_b


class ComplexDiv(nn.Module):
    """
    h = h_a + h_b i
    r = r_a + r_b i
    h / r = [(h_a * r_a + h_b * r_b) / (r_a ^2 + r_b ^2)] + [(h_b * r_a - h_a * r_b) / (r_a ^2 + r_b ^2)] i

    h in C^d
    r in C^d
    """

    def __init__(self):
        super(ComplexDiv, self).__init__()

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r

        r_norm = torch.sqrt(r_a ** 2 + r_b ** 2)

        t_a = (h_a * r_a + h_b * r_b) / r_norm
        t_b = (h_b * r_a - h_a * r_b) / r_norm
        return t_a, t_b

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

    def forward(self, h, r):
        h_a, h_b = h
        r_a, r_b = r
        a = self.complex_sub(self.complex_mul(h_a, r_a), self.complex_mul(h_b, self.complex_conj(r_b)))
        b = self.complex_add(self.complex_mul(h_a, r_b), self.complex_mul(h_b, self.complex_conj(r_a)))
        return a, b


# class QubitE(KGEModel):
#     def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
#         super().__init__()
#         self.num_entity = num_entity
#         self.num_relation = num_relation
#         self.hidden_dim = hidden_dim
#         self.epsilon = 2.0
#         self.p = p_norm
#
#         self.gamma = nn.Parameter(
#             torch.Tensor([gamma]),
#             requires_grad=False
#         )
#
#         self.embedding_range = nn.Parameter(
#             torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
#             requires_grad=False
#         )
#
#         self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 3))
#         nn.init.uniform_(
#             tensor=self.entity_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#         self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 4))
#         nn.init.uniform_(
#             tensor=self.relation_embedding,
#             a=-self.embedding_range.item(),
#             b=self.embedding_range.item()
#         )
#
#         # Initialize bias to 1
#         nn.init.ones_(
#             tensor=self.relation_embedding[:, 3 * hidden_dim:4 * hidden_dim]
#         )
#
#         self.pi = 3.14159262358979323846
#         self.mul = QubitMult()
#
#     def func(self, head, rel, tail, batch_type):
#         h_theta, h_phi, h_varphi = torch.chunk(head, 3, dim=2)
#         r_theta, r_phi, r_varphi, bias = torch.chunk(rel, 4, dim=2)
#         t_theta, t_phi, t_varphi = torch.chunk(tail, 3, dim=2)
#
#         bias = torch.abs(bias)
#
#         # Make phases of relations uniformly distributed in [-pi, pi]
#         h_theta = h_theta / (self.embedding_range.item() / self.pi)
#         h_phi = h_phi / (self.embedding_range.item() / self.pi)
#         h_varphi = h_varphi / (self.embedding_range.item() / self.pi)
#         cos_h_varphi = torch.cos(h_varphi)
#         sin_h_varphi = torch.sin(h_varphi)
#
#         # Obtain representation of the rotation axis
#         head_i = torch.cos(h_theta)
#         head_j = torch.sin(h_theta) * torch.cos(h_phi)
#         head_k = torch.sin(h_theta) * torch.sin(h_phi)
#         ha = cos_h_varphi
#         hai = sin_h_varphi * head_i
#         hb = sin_h_varphi * head_j
#         hbi = sin_h_varphi * head_k
#
#         # Make phases of relations uniformly distributed in [-pi, pi]
#         r_theta = r_theta / (self.embedding_range.item() / self.pi)
#         r_phi = r_phi / (self.embedding_range.item() / self.pi)
#         r_varphi = r_varphi / (self.embedding_range.item() / self.pi)
#         cos_r_varphi = torch.cos(r_varphi)
#         sin_r_varphi = torch.sin(r_varphi)
#
#         # Obtain representation of the rotation axis
#         rel_i = torch.cos(r_theta)
#         rel_j = torch.sin(r_theta) * torch.cos(r_phi)
#         rel_k = torch.sin(r_theta) * torch.sin(r_phi)
#
#         ra = cos_r_varphi
#         rai = sin_r_varphi * rel_i
#         rb = sin_r_varphi * rel_j
#         rbi = sin_r_varphi * rel_k
#
#         # Make phases of relations uniformly distributed in [-pi, pi]
#         t_theta = t_theta / (self.embedding_range.item() / self.pi)
#         t_phi = t_phi / (self.embedding_range.item() / self.pi)
#         t_varphi = t_varphi / (self.embedding_range.item() / self.pi)
#         cos_t_varphi = torch.cos(t_varphi)
#         sin_t_varphi = torch.sin(t_varphi)
#
#         # Obtain representation of the rotation axis
#         tail_i = torch.cos(t_theta)
#         tail_j = torch.sin(t_theta) * torch.cos(t_phi)
#         tail_k = torch.sin(t_theta) * torch.sin(t_phi)
#
#         ta = cos_t_varphi
#         tai = sin_t_varphi * tail_i
#         tb = sin_t_varphi * tail_j
#         tbi = sin_t_varphi * tail_k
#
#         h = ((ha, hai), (hb, hbi))
#         r = ((ra, rai), (rb, rbi))
#         (ta_, tai_), (tb_, tbi_) = self.mul(h, r)
#
#         score_a = ta * bias - ta_
#         score_ai = tai * bias - tai_
#         score_b = tb * bias - tb_
#         score_bi = tbi * bias - tbi_
#
#         score = torch.stack([score_a, score_ai, score_b, score_bi], dim=0)
#         score = score.norm(dim=0, p=self.p)
#         score = self.gamma.item() - score.sum(dim=2)
#
#         # Rotate the head entity
#         # uv = head_i * rel_i + head_j * rel_j + head_k * rel_k
#         #
#         # a = cos_r_varphi * cos_h_varphi - sin_r_varphi * sin_h_varphi * uv
#         #
#         # new_head_i = head_i * cos_r_varphi * sin_h_varphi + rel_i * cos_h_varphi * sin_r_varphi + sin_r_varphi * sin_h_varphi * (rel_j * head_k - head_j * rel_k)
#         # new_head_j = head_j * cos_r_varphi * sin_h_varphi + rel_j * cos_h_varphi * sin_r_varphi - sin_r_varphi * sin_h_varphi * (rel_i * head_k - head_i * rel_k)
#         # new_head_k = head_k * cos_r_varphi * sin_h_varphi + rel_k * cos_h_varphi * sin_r_varphi + sin_r_varphi * sin_h_varphi * (rel_i * head_j - head_i * rel_j)
#         #
#         # score_a = a * bias - cos_t_varphi
#         # score_i = new_head_i * bias - tail_i
#         # score_j = new_head_j * bias - tail_j
#         # score_k = new_head_k * bias - tail_k
#         #
#         # score = torch.stack([score_a, score_i, score_j, score_k], dim=0)
#         # score = score.norm(dim=0, p=self.p)
#         # score = self.gamma.item() - score.sum(dim=2)
#         return score


class RotatE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, p_norm):
        super().__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.p = p_norm

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = rel / (self.embedding_range.item() / self.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation - re_tail
        im_score = re_head * im_relation + im_head * re_relation - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0, p=self.p)
        score = self.gamma.item() - score.sum(dim=2)
        return score
