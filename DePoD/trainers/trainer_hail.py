from trainers.base import AbstractTrainer
import torch.nn.functional as F
from pathlib import Path
import torch
import numpy as np
import pickle

class Trainer(AbstractTrainer):
    def __init__(self, args, model, train_dataloader, test_dataloader, export_root):
        super().__init__(args, model, train_dataloader, test_dataloader, export_root)

        self.metric_ks = sorted(args.metric_ks)
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.selection = args.selection
        self.ablation = args.ablation
        self.num_elements = args.num_negative_elements + 1
        self.alpha = 1.0 if self.ablation == 'all' else args.alpha
        self.beta = args.beta
        self.gamma = args.gamma


        vocab_filename = Path().joinpath('datasets', args.dataset_code, 'vocab.pkl')
        user_history_filename = Path().joinpath('datasets', args.dataset_code, 'history.pkl')
        with open(user_history_filename, 'rb') as input_file:
            self.user_history = pickle.load(input_file)
        with open(vocab_filename, 'rb') as input_file:
            self.vocab = pickle.load(input_file)

        keys = self.vocab.counter.keys()
        values = self.vocab.counter.values()
        self.ids = self.vocab.convert_tokens_to_ids(keys)
        # normalize
        sum_value = np.sum([x for x in values])
        self.probability = [value / sum_value for value in values]


    def calculate_loss(self, batch_out, **kwargs):
        outs = batch_out['outs'],
        outs = torch.stack(outs, dim=0)
        masked_lm_ids = batch_out['masked_lm_ids']
        masked_lm_weights = batch_out['masked_lm_weights']

        masked_lm_ids = masked_lm_ids.view(-1)
        masked_lm_weights = masked_lm_weights.view(-1)
        one_hot_labels = F.one_hot(masked_lm_ids, outs.size(-1))

        logits = outs.view(outs.size(0), -1, outs.size(-1))
        ce_loss, top_order = self.ce_loss_peers(logits, one_hot_labels, masked_lm_weights)
        me_loss = self.me_loss_peers(logits, one_hot_labels, masked_lm_weights, top_order)

        total_loss = self.alpha*ce_loss + (1 - self.alpha) * me_loss
        return total_loss

    # mutual exclusivity loss for multiple models
    def me_loss_peers(self, logits, one_hot_labels, masked_lm_weights, top_order):
        if self.ablation == 'all':
            return 0

        num_peers = logits.size(0)

        probs = F.softmax(logits, -1)
        me_probs = 1 - probs
        log_probs = F.log_softmax(logits, -1)
        log_me_probs = self.log_me_softmax(logits, -1)

        denominator = torch.sum(masked_lm_weights) + 1e-5

        me_loss = 0
        for peer_id in range(num_peers):
            peer_list = torch.arange(num_peers).to(self.device)
            arr1 = peer_list[0:peer_id]
            arr2 = peer_list[peer_id+1:]
            peer_list = torch.cat((arr1, arr2), dim=0)

            me_weight = torch.index_select(me_probs, dim=0, index=peer_list).detach()
            me_weight = torch.pow(me_weight, self.gamma)

            weight = torch.index_select(probs, dim=0, index=peer_list).detach()
            weight = torch.pow(weight, self.gamma)

            loss_positive = one_hot_labels * me_weight * log_probs[peer_id]
            loss_positive = torch.mean(loss_positive, 0)
            loss_negative = (1 - one_hot_labels) * weight * log_me_probs[peer_id]
            loss_negative = torch.mean(loss_negative, 0)

            if self.ablation == 'original':
                loss = loss_positive + loss_negative
            elif self.ablation == 'positive':
                loss = loss_negative
            elif self.ablation == 'negative':
                loss = loss_positive

            loss = masked_lm_weights * (-torch.sum(loss, -1))

            numerator = torch.sum(top_order*loss, -1)
            me_loss += numerator / denominator

        return me_loss

    def log_me_softmax(self, x, index=-1):
        # return x - x.exp().sum(-1).log().unsqueeze(-1)
        x = x.double()
        numerator = (x.exp().sum(index).unsqueeze(index) - x.exp()).log()
        res = numerator - x.exp().sum(index).log().unsqueeze(index)
        return res.float()

    # cross entropy loss for multiple models
    def ce_loss_peers(self, logits, one_hot_labels, masked_lm_weights):
        logits = logits
        log_probs = F.log_softmax(logits, -1)
        loss = one_hot_labels * log_probs
        loss = masked_lm_weights * (-torch.sum(loss, -1))

        top_order = self.search_truncation(loss) if self.beta > 0 else 1

        numerator = torch.sum(top_order*loss, -1)     #top_order*
        denominator = torch.sum(masked_lm_weights) + 1e-5
        loss = numerator/denominator
        return torch.sum(loss, -1), top_order

    def ce_loss_peers(self, logits, one_hot_labels, masked_lm_weights):
        log_probs = F.log_softmax(logits, -1)
        loss = one_hot_labels * log_probs
        loss = masked_lm_weights * (-torch.sum(loss, -1))

        shape_number = logits.size(1)
        top_n = shape_number * 0.01
        top_order = torch.tensor(True)
        for lo in loss:
            order = lo.topk(shape_number, largest=True, sorted=True)[1].topk(shape_number, largest=False, sorted=True)[1]
            top_order = torch.logical_and(order < top_n, top_order)

        top_order = torch.logical_not(top_order)

        numerator = torch.sum(loss, -1)
        denominator = torch.sum(masked_lm_weights) + 1e-5
        loss = numerator/denominator
        return loss, top_order

    def search_truncation(self, all_loss):
        shape_number = all_loss.size(1)
        num_peers = all_loss.size(0)
        top_n = shape_number * (1 - self.beta)
        top_order = None
        flag = False
        for model_id in range(num_peers):
            loss = all_loss[model_id]
            order = loss.argsort().argsort()

            top_order = (order > top_n)&top_order if flag else (order > top_n)
            flag = True

        top_order = ~top_order
        return top_order

    def calculate_metrics(self, batch_out, average_meter_set):
        outs = batch_out['outs']
        input_ids = batch_out['input_ids']
        masked_lm_ids = batch_out['masked_lm_ids']
        info = batch_out['info']

        outs = torch.stack(outs, dim=0)
        num_peers = outs.size(0)
        for model_id in range(num_peers):
            logits = outs[0]
            average_meter_set = self.calculate_metrics_one_model(logits, input_ids, masked_lm_ids, info, model_id, average_meter_set)
        return average_meter_set

    def calculate_metrics_one_model(self, logits, input_ids, masked_lm_ids, info, model_id, average_meter_set):
        masked_lm_probs = F.softmax(logits, -1)
        masked_lm_probs = masked_lm_probs.view((-1, self.max_predictions_per_seq, masked_lm_probs.shape[1]))

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx].cpu().numpy().tolist())
            rated.add(0)
            rated.add(masked_lm_ids[idx][0].item())
            list(map(lambda x: rated.add(x), self.user_history["user_" + str(info[idx][0].item())][0]))
            item_idx = [masked_lm_ids[idx][0].item()]
            masked_lm_probs_elem = masked_lm_probs[idx][0]

            probability = self.probability if self.selection == 'popular' else None
            while len(item_idx) < self.num_elements:
                sampled_ids = np.random.choice(self.ids, self.num_elements, replace=False, p=probability)
                sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                item_idx.extend(sampled_ids[:])
            item_idx = item_idx[:self.num_elements]

            predictions = -masked_lm_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0].item()

            for k in self.metric_ks:
                value = 1 if rank < k else 0
                average_meter_set.update('M%s_hit_%s'%(model_id, k), value)
                average_meter_set.update('M%s_ndcg_%s'%(model_id, k), value / np.log2(rank + 2))
            average_meter_set.update('M%s_ap'%model_id, 1.0 / (rank + 1))

        return average_meter_set

    @classmethod
    def code(cls):
        return 'Trainer'
