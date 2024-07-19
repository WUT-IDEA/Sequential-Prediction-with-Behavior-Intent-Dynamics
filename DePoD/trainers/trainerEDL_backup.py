from trainers.base import AbstractTrainer
import torch.nn.functional as F
from pathlib import Path
import torch
import numpy as np
import pickle

class TrainerEDL(AbstractTrainer):
    def __init__(self, args, model, train_dataloader, test_dataloader, export_root):
        super().__init__(args, model, train_dataloader, test_dataloader, export_root)

        self.metric_ks = sorted(args.metric_ks)
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.selection = args.selection
        self.loss_term = args.loss_term
        self.num_elements = args.num_negative_elements + 1
        # self.alpha = 1.0 if self.ablation == 'all' else args.alpha
        #         # self.beta = args.beta
        #         # self.gamma = args.gamma


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


    def calculate_loss(self, batch_out, current_epoch):
        outs = batch_out['outs']
        outs = torch.stack(outs, dim=0)
        masked_lm_ids = batch_out['masked_lm_ids']
        masked_lm_weights = batch_out['masked_lm_weights']

        masked_lm_ids = masked_lm_ids.view(-1)
        masked_lm_weights = masked_lm_weights.view(-1)
        one_hot_labels = F.one_hot(masked_lm_ids, outs.size(-1))

        logits = outs.view(outs.size(0), -1, outs.size(-1))

        evidence = self.logits2evidence(logits, evidence_type='exp')
        alpha = evidence + 1

        # K = outs.size(-1)
        # sum_alpha = torch.sum(alpha, axis=-1, keepdim=True)
        # u = K / sum_alpha  # uncertainty
        # prob = alpha / sum_alpha

        total_loss = 0
        if 'mse' in self.loss_term:
            mse_loss = self.mse_loss_peers(one_hot_labels, masked_lm_weights, alpha)
            total_loss += mse_loss
        if 'kl' in self.loss_term:
            kl_loss = self.kl_loss(one_hot_labels, masked_lm_weights, alpha, current_epoch)
            total_loss += kl_loss
        #cross-entropy
        if 'ce' in self.loss_term:
            ce_loss = self.risk_loss(one_hot_labels, masked_lm_weights, alpha, func=torch.digamma)
            total_loss += ce_loss
        # negated logarithm
        if 'ml' in self.loss_term:
            ml_loss = self.risk_loss(one_hot_labels, masked_lm_weights, alpha, func=torch.log)
            total_loss += ml_loss
        return total_loss

    def risk_loss(self, one_hot_labels, masked_lm_weights, alpha, func):
        S = torch.sum(alpha, axis=-1, keepdim=True)
        risk_loss = torch.sum(one_hot_labels * (func(S) - func(alpha)), axis=-1)

        numerator = masked_lm_weights * risk_loss
        denominator = torch.sum(masked_lm_weights) + 1e-5
        risk_loss = numerator / denominator
        return torch.sum(risk_loss)

    def kl_loss(self, one_hot_labels, masked_lm_weights, alpha, current_epoch):
        E = alpha - 1
        # alp is the Dirichlet parameters after removal of the non-misleading evidence from predicted parameters alpha
        alp = E * (1 - one_hot_labels) + 1
        beta = torch.ones_like(alp)

        kl_loss = self.Dirichlet_KL(alp, beta)
        kl_loss = kl_loss.reshape([kl_loss.size(0), -1])
        numerator = masked_lm_weights * kl_loss
        denominator = torch.sum(masked_lm_weights) + 1e-5
        kl_loss = numerator / denominator

        annealing_coef = np.minimum(1.0, current_epoch / 10)
        kl_loss = annealing_coef * torch.sum(kl_loss)
        return kl_loss

    # cross entropy loss for multiple models
    def mse_loss_peers(self, one_hot_labels, masked_lm_weights, alpha):
        S = torch.sum(alpha, axis=-1, keepdim=True)
        expected_prob= alpha / S

        err = torch.sum((one_hot_labels - expected_prob)**2, axis=-1)
        var = torch.sum((expected_prob*(1 - expected_prob))/(S + 1), axis=-1)

        numerator = masked_lm_weights * (err + var)
        denominator = torch.sum(masked_lm_weights) + 1e-5
        loss = numerator / denominator
        return torch.sum(loss)

    def calculate_metrics(self, batch_out, average_meter_set):
        outs = batch_out['outs']
        input_ids = batch_out['input_ids']
        masked_lm_ids = batch_out['masked_lm_ids']
        info = batch_out['info']

        K = outs[0].size(-1)

        outs = torch.stack(outs, dim=0)
        num_peers = outs.size(0)
        for model_id in range(num_peers):
            logits = outs[model_id]

            evidence = self.logits2evidence(logits, evidence_type='exp')
            alpha = evidence + 1
            sum_alpha = torch.sum(alpha, axis=-1, keepdim=True)
            u = K / sum_alpha  # uncertainty
            masked_lm_probs = alpha / sum_alpha

            average_meter_set = self.calculate_metrics_one_model(masked_lm_probs, input_ids, masked_lm_ids, info, model_id, average_meter_set)
        return average_meter_set

    def calculate_metrics_one_model(self, masked_lm_probs, input_ids, masked_lm_ids, info, model_id, average_meter_set):
        masked_lm_probs = masked_lm_probs.view((-1, self.max_predictions_per_seq, masked_lm_probs.shape[1]))

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx].cpu().numpy().tolist())
            rated.add(0)
            rated.add(masked_lm_ids[idx][0].item())
            list(map(lambda x: rated.add(x), self.user_history["user_" + str(info[idx][0].item())][0]))
            item_idx = [masked_lm_ids[idx][0].item()]
            masked_lm_probs_elem = masked_lm_probs[idx][0]

            probability = self.probability if self.selection is 'popular' else None
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

    def logits2evidence(self, logits, evidence_type='exp'):
        if evidence_type=='relu':
            # This function to generate evidence is used for the first example
            return F.relu(logits)
        elif evidence_type=='exp':
            # This one usually works better and used for the second and third examples
            # For general settings and different datasets, you may try this one first
            return torch.exp(logits)
        elif evidence_type=='softplus':
            # This one is another alternative and
            # usually behaves better than the relu_evidence
            return F.softplus(logits)

    def Dirichlet_KL(self, alpha, beta):
        S_alpha = torch.sum(alpha, axis=-1, keepdim=True)
        S_beta = torch.sum(beta, axis=-1, keepdim=True)

        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), axis=-1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), axis=-1, keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), axis=-1, keepdim=True) + lnB + lnB_uni
        return kl


    @classmethod
    def code(cls):
        return 'EvidenceTrainer'