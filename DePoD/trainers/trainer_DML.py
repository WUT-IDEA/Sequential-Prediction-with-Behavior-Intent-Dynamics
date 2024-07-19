from trainers.base import AbstractTrainer
import torch.nn.functional as F
from pathlib import Path
import torch
import numpy as np
import pickle
from torch.utils.data.sampler import WeightedRandomSampler

class Trainer(AbstractTrainer):
    def __init__(self, args, model, train_dataloader, test_dataloader, export_root):
        super().__init__(args, model, train_dataloader, test_dataloader, export_root)

        self.metric_ks = sorted(args.metric_ks)
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.selection = args.selection
        self.non_tartget_sampling_strategy = args.non_tartget_sampling_strategy
        self.num_elements = args.num_negative_elements + 1
        self.num_epochs = args.num_epochs

        self.learning_pattern = args.learning_pattern
        self.alpha = args.alpha
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
        outs = batch_out['outs'].transpose(0, 1)
        tea_score = batch_out['tea_score']
        tea_score = tea_score if tea_score is None else tea_score.transpose(0,1)
        current_epoch = kwargs['current_epoch']

        masked_lm_ids = batch_out['masked_lm_ids']
        masked_lm_weights = batch_out['masked_lm_weights']

        masked_lm_ids = masked_lm_ids.view(-1)
        masked_lm_weights = masked_lm_weights.view(-1)
        one_hot_labels = F.one_hot(masked_lm_ids, outs.size(-1))

        logits = outs.view(outs.size(0), -1, outs.size(-1)) #
        num_peers = logits.size(0)

        ce_loss, top_order = self.ce_loss_peers(logits, one_hot_labels, masked_lm_weights)
        loss = ce_loss
        if self.learning_pattern != 'separate' and self.base_models_num > 1:
            kd_loss = self.kd_loss_peers(logits, tea_score, one_hot_labels, masked_lm_weights, current_epoch, top_order)
            loss += kd_loss
        return loss.sum()

    def kd_loss_peers(self, logits, tea_score, one_hot_labels, masked_lm_weights, current_epoch, top_order):
        num_peers = logits.size(0)

        non_target_loss = []
        target_loss = []
        t = current_epoch / self.num_epochs

        nckd_logits = logits - 1000.0 * one_hot_labels
        nckd_log_probs = F.log_softmax(nckd_logits, -1)

        for peer_id in range(num_peers):
            if tea_score is None:
                tea_weight=1
            else:
                tea_score1 = tea_score[0:peer_id]
                tea_score2 = tea_score[peer_id + 1:]
                tea_weight = torch.cat((tea_score1, tea_score2), dim=0)
                tau = (1-t)+1e-5
                tea_weight = F.gumbel_softmax(tea_weight, tau=tau, hard=True, dim=0)

            if self.learning_pattern in ['mutual_target', 'mutual_all','mutual_target_alpha', 'mutual_target_alpha_nontarget','mutual_target_nontarget_one', 'DKD']:
                # target_loss
                arr1 = logits[0:peer_id]
                arr2 = logits[peer_id + 1:]
                tea_target_logit = (torch.cat((arr1, arr2), dim=0).detach() * tea_weight).sum(dim=0)#.mean(dim=0)

                tea_target_prob = (one_hot_labels * F.softmax(tea_target_logit, -1)).sum(-1)
                stu_target_prob = (one_hot_labels * F.softmax(logits[peer_id], -1)).sum(-1)

                target_numerator = self.alpha * (    tea_target_prob * torch.log(stu_target_prob) + (1-tea_target_prob)*torch.log(1-stu_target_prob)   )
                target_numerator = masked_lm_weights * (-target_numerator)
                target_numerator = torch.sum(target_numerator, dim=-1, keepdim=True)
                denominator = torch.sum(masked_lm_weights) + 1e-5
                target_loss.append(target_numerator / denominator)

            if self.learning_pattern in ['mutual_nontarget', 'mutual_all', 'mutual_target_alpha_nontarget', 'mutual_target_nontarget_one', 'DKD']:
                # non_target_loss
                arr1 = nckd_logits[0:peer_id]
                arr2 = nckd_logits[peer_id + 1:]
                tea_nckd_logit = (torch.cat((arr1, arr2), dim=0).detach() * tea_weight).sum(dim=0)#.mean(dim=0)

                tea_nckd_prob = F.softmax(tea_nckd_logit, -1)#.detach()
                stu_nckd_log_prob = nckd_log_probs[peer_id]

                nckd_numerator = tea_nckd_prob * stu_nckd_log_prob
                nckd_numerator = masked_lm_weights * (-torch.sum(nckd_numerator, dim=-1))
                nckd_numerator = torch.sum(nckd_numerator, dim=-1, keepdim=True)
                denominator = torch.sum(masked_lm_weights) + 1e-5
                non_target_loss.append(nckd_numerator / denominator)

        target_loss = torch.cat(target_loss, dim=0)
        non_target_loss = torch.cat(non_target_loss, dim=0)

        kd_loss = target_loss + self.beta * non_target_loss
        return kd_loss


    # cross entropy loss for multiple models
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

    def calculate_metrics(self, batch_out, average_meter_set):
        outs = batch_out['outs'].transpose(0,1)
        input_ids = batch_out['input_ids']
        masked_lm_ids = batch_out['masked_lm_ids']
        info = batch_out['info']

        num_peers = outs.size(0)
        for model_id in range(num_peers):
            logits = outs[model_id]
            masked_lm_probs = F.softmax(logits, -1)
            average_meter_set = self.calculate_metrics_one_model(masked_lm_probs, input_ids, masked_lm_ids, info, model_id, average_meter_set)
        return average_meter_set

    def calculate_metrics_one_model(self, masked_lm_probs, input_ids, masked_lm_ids, info, model_id, average_meter_set):
        masked_lm_probs = masked_lm_probs.view((-1, self.max_predictions_per_seq, masked_lm_probs.shape[1]))

        for idx in range(len(input_ids)):
            item_idx = [masked_lm_ids[idx][0].item()]
            masked_lm_probs_elem = masked_lm_probs[idx][0]

            if self.num_elements > 0:
                rated = set(input_ids[idx].cpu().numpy().tolist())
                rated.add(0)
                rated.add(masked_lm_ids[idx][0].item())
                list(map(lambda x: rated.add(x), self.user_history["user_" + str(info[idx][0].item())][0]))

                probability = self.probability if self.selection == 'popular' else None
                while len(item_idx) < self.num_elements:
                    sampled_ids = np.random.choice(self.ids, self.num_elements, replace=False, p=probability)
                    sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                    item_idx.extend(sampled_ids[:])
                item_idx = item_idx[:self.num_elements]
            else:
                item_idx = self.ids

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
