from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from QubitE_good import QubitE
from toolbox.data.DataSchema import RelationalTripletData, RelationalTripletDatasetCachePath
from toolbox.data.DatasetSchema import get_dataset
from toolbox.data.LinkPredictDataset import LinkPredictDataset
from toolbox.data.ScoringAllDataset import ScoringAllDataset
from toolbox.data.functional import with_inverse_relations, build_map_hr_t
from toolbox.evaluate.Evaluate import get_score
from toolbox.evaluate.LinkPredict import batch_link_predict2, as_result_dict2, batch_link_predict_type_constraint
from toolbox.exp.Experiment import Experiment
from toolbox.exp.OutputSchema import OutputSchema
from toolbox.optim.lr_scheduler import get_scheduler
from toolbox.utils.Progbar import Progbar
from toolbox.utils.RandomSeeds import set_seeds


class MyExperiment(Experiment):

    def __init__(self, output: OutputSchema, data: RelationalTripletData,
                 start_step, max_steps, every_test_step, every_valid_step,
                 batch_size, test_batch_size, sampling_window_size, label_smoothing,
                 train_device, test_device,
                 resume, resume_by_score,
                 lr, amsgrad, lr_decay, weight_decay,
                 edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
                 ):
        super(MyExperiment, self).__init__(output)
        data.load_cache(["train_triples_ids", "test_triples_ids", "valid_triples_ids", "all_triples_ids"])
        data.load_cache(["hr_t_train"])
        data.print(self.log)
        self.store.save_scripts(["train_QubitE.py", "QubitE_good.py", "QubitEmbedding.py"])
        max_relation_id = data.relation_count

        # 1. build train dataset
        train_triples, _, _ = with_inverse_relations(data.train_triples_ids, max_relation_id)
        self.head_type_constraint = defaultdict(list)
        self.tail_type_constraint = defaultdict(list)
        for h, r, t in train_triples:
            self.head_type_constraint[r].append(h)
            self.tail_type_constraint[r].append(t)
            self.head_type_constraint[r + max_relation_id].append(t)
            self.tail_type_constraint[r + max_relation_id].append(h)
        self.entity_count = data.entity_count
        train_data = ScoringAllDataset(train_triples, data.entity_count)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # 2. build valid and test dataset
        all_triples, _, _ = with_inverse_relations(data.all_triples_ids, max_relation_id)
        hr_t = build_map_hr_t(all_triples)
        valid_data = LinkPredictDataset(data.valid_triples_ids, hr_t, max_relation_id, data.entity_count)
        test_data = LinkPredictDataset(data.test_triples_ids, hr_t, max_relation_id, data.entity_count)
        valid_dataloader = DataLoader(valid_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        # 3. build model
        model = QubitE(data.entity_count, 2 * data.relation_count, edim).to(train_device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
        scheduler = get_scheduler(opt, lr_policy="step")
        best_score = 0
        if resume:
            if resume_by_score > 0:
                start_step, _, best_score = self.store.load_by_score(model, opt, resume_by_score)
            else:
                start_step, _, best_score = self.store.load_best(model, opt)
            self.dump_model(model)
            model.eval()
            with torch.no_grad():
                self.debug("Resumed from score %.4f." % best_score)
                self.debug("Take a look at the performance after resumed.")
                self.debug("Validation (step: %d):" % start_step)
                self.evaluate(model, valid_data, valid_dataloader, test_batch_size, max_relation_id, test_device)
                self.debug("Test (step: %d):" % start_step)
                self.evaluate(model, test_data, test_dataloader, test_batch_size, max_relation_id, test_device)
        else:
            model.init()
            self.dump_model(model)

        # 4. training
        self.debug("training")
        progbar = Progbar(max_step=max_steps)
        for step in range(start_step, max_steps):
            model.train()
            losses = []
            for h, r, targets in train_dataloader:
                opt.zero_grad()

                h = h.to(train_device)
                r = r.to(train_device)
                targets = targets.to(train_device).float()
                if label_smoothing:
                    targets = ((1.0 - label_smoothing) * targets) + (1.0 / targets.size(1))

                predictions = model(h, r)
                loss = model.loss(predictions, targets)
                # loss = loss + model.regular_loss(h, r)
                losses.append(loss.item())
                loss.backward()
                opt.step()
            scheduler.step()

            progbar.update(step + 1, [("step", step + 1), ("loss", torch.mean(torch.Tensor(losses)).item()), ("lr", torch.mean(torch.Tensor(scheduler.get_last_lr())).item())])
            if (step + 1) % every_valid_step == 0:
                print("")
                model.eval()
                with torch.no_grad():
                    self.debug("Validation (step: %d):" % (step + 1))
                    result = self.evaluate(model, valid_data, valid_dataloader, test_batch_size, max_relation_id, test_device)
                    self.visual_result(step + 1, result, "Valid-")
                    score = get_score(result)
                    self.store.save_by_score(model, opt, step, 0, score)
                    if score >= best_score:
                        self.success("current score=%.4f > best score=%.4f" % (score, best_score))
                        best_score = score
                        self.debug("saving best score %.4f" % score)
                        self.store.save_best(model, opt, step, 0, score)
                    else:
                        self.fail("current score=%.4f < best score=%.4f" % (score, best_score))
                print("")
            if (step + 1) % every_test_step == 0:
                print("")
                model.eval()
                with torch.no_grad():
                    self.debug("Test (step: %d):" % (step + 1))
                    result = self.evaluate(model, test_data, test_dataloader, test_batch_size, max_relation_id, test_device)
                    self.visual_result(step + 1, result, "Test-")
                print("")

    def evaluate(self, model, test_data, test_dataloader, test_batch_size, max_relation_id: int, device="cuda:0"):
        data = iter(test_dataloader)

        def predict(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model(h, r)
            pred_head = model(t, reverse_r)
            pred_tail = (pred_tail[0] + pred_tail[1]) / 2
            pred_head = (pred_head[0] + pred_head[1]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser

        progbar = Progbar(max_step=len(test_data) // (test_batch_size * 5))

        def log(i, hits, hits_left, hits_right, ranks, ranks_left, ranks_right):
            if i % (test_batch_size * 5) == 0:
                progbar.update(i // (test_batch_size * 5), [("Hits @10", np.mean(hits[9]))])

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict2(test_batch_size, len(test_data), predict, log)
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))

        self.log("with type constraint")
        data = iter(test_dataloader)

        def predict_type_constraint(i):
            h, r, mask_for_hr, t, reverse_r, mask_for_tReverser = next(data)
            h = h.to(device)
            r = r.to(device)
            mask_for_hr = mask_for_hr.to(device)
            t = t.to(device)
            reverse_r = reverse_r.to(device)
            mask_for_tReverser = mask_for_tReverser.to(device)
            pred_tail = model(h, r)
            pred_head = model(t, reverse_r)
            pred_tail = (pred_tail[0] + pred_tail[1]) / 2
            pred_head = (pred_head[0] + pred_head[1]) / 2
            return t, h, pred_tail, pred_head, mask_for_hr, mask_for_tReverser, r, reverse_r

        hits, hits_left, hits_right, ranks, ranks_left, ranks_right = batch_link_predict_type_constraint(
            self.entity_count, self.head_type_constraint, self.tail_type_constraint, test_batch_size, len(test_data), predict_type_constraint, log
        )
        result = as_result_dict2((hits, hits_left, hits_right, ranks, ranks_left, ranks_right))
        for i in (0, 2, 9):
            self.log('Hits @{0:2d}: {1:2.2%}    left: {2:2.2%}    right: {3:2.2%}'.format(i + 1, np.mean(hits[i]), np.mean(hits_left[i]), np.mean(hits_right[i])))
        self.log('Mean rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(ranks), np.mean(ranks_left), np.mean(ranks_right)))
        self.log('Mean reciprocal rank: {0:.3f}    left: {1:.3f}    right: {2:.3f}'.format(np.mean(1. / np.array(ranks)), np.mean(1. / np.array(ranks_left)), np.mean(1. / np.array(ranks_right))))

        return result

    def visual_result(self, step_num: int, result, scope: str):
        average = result["average"]
        left2right = result["left2right"]
        right2left = result["right2left"]
        sorted(average)
        sorted(left2right)
        sorted(right2left)
        for i in average:
            self.vis.add_scalar(scope + i, average[i], step_num)
        for i in left2right:
            self.vis.add_scalar(scope + i, left2right[i], step_num)
        for i in right2left:
            self.vis.add_scalar(scope + i, right2left[i], step_num)

    def dump_model(self, model):
        self.debug(model)
        self.debug("")
        self.debug("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.debug(name)
        self.debug("")


@click.command()
@click.option("--dataset", type=str, default="FB15k-237", help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
@click.option("--name", type=str, default="QubitE", help="Name of the experiment.")
@click.option("--start_step", type=int, default=0, help="start step.")
@click.option("--max_steps", type=int, default=1000, help="Number of steps.")
@click.option("--every_test_step", type=int, default=10, help="Number of steps.")
@click.option("--every_valid_step", type=int, default=5, help="Number of steps.")
@click.option("--batch_size", type=int, default=512, help="Batch size.")
@click.option("--test_batch_size", type=int, default=512, help="Test batch size.")
@click.option("--sampling_window_size", type=int, default=1000, help="Sampling window size.")
@click.option("--label_smoothing", type=float, default=0.1, help="Amount of label smoothing.")
@click.option("--train_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--test_device", type=str, default="cuda:0", help="choice: cuda:0, cuda:1, cpu.")
@click.option("--resume", type=bool, default=False, help="Resume from output directory.")
@click.option("--resume_by_score", type=float, default=0.0, help="Resume by score from output directory. Resume best if it is 0. Default: 0")
@click.option("--lr", type=float, default=0.003, help="Learning rate.")
@click.option("--amsgrad", type=bool, default=False, help="AMSGrad for Adam.")
@click.option("--lr_decay", type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
@click.option('--weight_decay', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
@click.option("--edim", type=int, default=200, help="Entity embedding dimensionality.")
@click.option("--rdim", type=int, default=200, help="Relation embedding dimensionality.")
@click.option("--input_dropout", type=float, default=0.2, help="Input layer dropout.")
@click.option("--hidden_dropout1", type=float, default=0.2, help="Dropout after the first hidden layer.")
@click.option("--hidden_dropout2", type=float, default=0.2, help="Dropout after the second hidden layer.")
def main(dataset, name,
         start_step, max_steps, every_test_step, every_valid_step,
         batch_size, test_batch_size, sampling_window_size, label_smoothing,
         train_device, test_device,
         resume, resume_by_score,
         lr, amsgrad, lr_decay, weight_decay,
         edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
         ):
    set_seeds()
    output = OutputSchema(dataset + "-" + name)
    data_home = Path.home() / "data"
    if dataset == "all":
        datasets = [get_dataset(i, data_home) for i in ["FB15k", "FB15k-237", "WN18", "WN18RR"]]
    else:
        datasets = [get_dataset(dataset, data_home)]

    for i in datasets:
        dataset = i
        cache = RelationalTripletDatasetCachePath(dataset.cache_path)
        data = RelationalTripletData(dataset=dataset, cache_path=cache)
        data.preprocess_data_if_needed()
        data.load_cache(["meta"])

        MyExperiment(
            output, data,
            start_step, max_steps, every_test_step, every_valid_step,
            batch_size, test_batch_size, sampling_window_size, label_smoothing,
            train_device, test_device,
            resume, resume_by_score,
            lr, amsgrad, lr_decay, weight_decay,
            edim, rdim, input_dropout, hidden_dropout1, hidden_dropout2,
        )


if __name__ == '__main__':
    main()
