from trainers.base import AbstractTrainer
import torch.nn.functional as F
from pathlib import Path
import torch
import numpy as np
import pickle
from utils import AverageMeterSet
from tqdm import tqdm
from .AutomaticWeightedLoss import AutomaticWeightedLoss
# from sklearn.metrics import roc_auc_score
import torch.optim as optim

class FinetuningEDL(AbstractTrainer):
    def __init__(self, args, model, train_dataloader, test_dataloader, export_root):
        super().__init__(args, model, train_dataloader, test_dataloader, export_root)

        self.metric_ks = sorted(args.metric_ks)
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.selection = args.selection
        self.loss_term = args.loss_term.split('+')
        self.num_elements = args.num_negative_elements + 1
        self.eps = args.off_mainfold_eps

        self.awl = AutomaticWeightedLoss(len(self.loss_term ))
        # set param_optimizer
        self.optimizer_w = optim.Adam([{'params': self.awl.parameters(), 'weight_decay': 0}], lr=0.001)

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

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights, corrupt_tokens = batch

            batch_input = {
                'info': info,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_ids': masked_lm_ids,
                'masked_lm_weights': masked_lm_weights,
                'corrupt_tokens': corrupt_tokens
            }

            off_manifold_x = self.off_manifold_generate(batch_input, rand_init=True)

            total_loss = []

            self.model.train()
            # first learning for accuracy

            batch_out = self.model(batch_input)
            seq_embeddings_ori = batch_out['seq_embeddings']
            prob_ori, u_ori = self.calculate_output(batch_out)

            self.optimizer.zero_grad()
            loss_ml = self.calculate_loss(batch_out)
            total_loss.append(loss_ml)
            # loss_ml.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            # self.optimizer.step_and_update_lr()

            # second learning for OOD
            if ('cl' in self.loss_term) or ('uncertainty' in self.loss_term):
                batch_out_off = self.model(batch_input, off_manifold_x)
                prob_off, u_off = self.calculate_output(batch_out_off)
                seq_embeddings_off = batch_out_off['seq_embeddings']

                batch_out_corrupt = self.model(batch_input, batch_input['corrupt_tokens'])
                prob_corrupt, u_corrupt = self.calculate_output(batch_out_corrupt)
                seq_embeddings_corrupt = batch_out_corrupt['seq_embeddings']


            if 'cl' in self.loss_term:
                loss_cl = self.contrastive_loss(seq_embeddings_ori, seq_embeddings_off, seq_embeddings_corrupt,
                                                batch_out['masked_lm_weights'])
                total_loss.append(loss_cl)

            if 'uncertainty' in self.loss_term:
                loss_uncertainy = self.uncertainy_loss(u_ori, u_corrupt, u_off, beta_ori=1, beta_corrupt=1, beta_off=1,
                                                       masked_lm_weights=batch_out['masked_lm_weights'])
                total_loss.append(loss_uncertainy)

            # loss_2 = loss_ml + 2*loss_uncertainy + 10*loss_cl
            loss = self.awl(total_loss)

            self.optimizer_w.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step_and_update_lr()
            self.optimizer_w.step()

            # end learning

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f}'.format(epoch+1, average_meter_set['loss'].avg))

        tqdm_dataloader.set_description('Logging to Tensorboard')
        log_data = {
            'state_dict': (self._create_state_dict()),
            'epoch': epoch+1
        }

        if self.log_test:
            self.record_test_result(log_data, epoch)

        log_data.update(average_meter_set.averages())
        self.logger_service.log_train(log_data)     #save model and loss after one epoch

    def uncertainy_loss(self, u_ori, u_corrupt, u_off, beta_ori, beta_corrupt, beta_off, masked_lm_weights):
        masked_lm_weights = masked_lm_weights.view(-1)
        loss_corrupt = masked_lm_weights * -torch.log(1 - u_corrupt.squeeze(-1))
        loss_ori = masked_lm_weights * -torch.log(1 - u_ori.squeeze(-1))
        loss_off = masked_lm_weights * -torch.log(u_off.squeeze(-1))

        numerator = beta_ori * torch.sum(loss_ori, -1) + beta_corrupt * torch.sum(loss_corrupt, -1) + beta_off * torch.sum(loss_off, -1)

        denominator = torch.sum(masked_lm_weights) + 1e-5
        loss_uncertainy = numerator / denominator
        return torch.sum(loss_uncertainy)


    def contrastive_loss(self, embeddings_ori, embeddings_off, embeddings_corrupt, masked_lm_weights):
        masked_lm_weights = masked_lm_weights.view(-1)

        ori_corrupt_score = torch.cosine_similarity(embeddings_ori, embeddings_corrupt, dim=-1)
        ori_corrupt_score = torch.exp(ori_corrupt_score)

        ori_off_score = torch.cosine_similarity(embeddings_ori, embeddings_off, dim=-1)
        ori_off_score = torch.exp(ori_off_score)

        numerator = ori_corrupt_score / (ori_corrupt_score + ori_off_score)
        numerator = -torch.log(numerator)
        denominator = torch.sum(masked_lm_weights) + 1e-5
        loss_cl = torch.sum(numerator, -1) / denominator
        return torch.sum(loss_cl)

    def similarity_sample(self, mat1, mat2):
        mat1 = mat1.unsqueeze(1)
        mat2 = mat2.unsqueeze(2)

        similarity_score = torch.exp(torch.bmm(mat1, mat2))
        return similarity_score

    def off_manifold_generate(self, batch_input, rand_init=False):
        input_ids = batch_input['input_ids']

        self.model.eval()
        with torch.no_grad():
            if self.is_parallel:
                embedding = self.model.module.get_input_embeddings()(input_ids)
            else:
                embedding = self.model.get_input_embeddings()(input_ids)

        input_embedding = embedding.detach()

        input_mask = batch_input['input_mask']
        mask = input_mask.unsqueeze(-1).repeat(1, 1, input_embedding.size(-1))
        # random init the adv samples
        if rand_init:
            input_embedding = input_embedding + mask*torch.zeros_like(input_embedding).uniform_(-self.eps, self.eps)

        input_embedding.requires_grad = True

        if input_embedding.grad is not None:
            input_embedding.grad.data.fill_(0)

        batch_out = self.model(batch_input, input_embedding)
        cost = self.calculate_loss(batch_out)

        self.model.zero_grad()
        cost.backward()

        off_samples = input_embedding + self.eps * torch.sign(input_embedding.grad.data*mask)
        off_samples = torch.min(torch.max(off_samples, embedding - self.eps), embedding + self.eps)

        self.model.train()
        return off_samples.detach()

    def calculate_output(self, batch_out):
        outs = batch_out['outs']

        logits = outs.view(outs.size(0), -1, outs.size(-1))

        evidence = self.logits2evidence(logits, evidence_type='exp')
        alpha = evidence + 1
        K = outs.size(-1)
        sum_alpha = torch.sum(alpha, axis=-1, keepdim=True)
        u = K / sum_alpha  # uncertainty
        prob = alpha / sum_alpha
        return prob, u

    def calculate_loss(self, batch_out, **kwargs):
        outs = batch_out['outs']

        masked_lm_ids = batch_out['masked_lm_ids']
        masked_lm_weights = batch_out['masked_lm_weights']

        masked_lm_ids = masked_lm_ids.view(-1)
        masked_lm_weights = masked_lm_weights.view(-1)
        one_hot_labels = F.one_hot(masked_lm_ids, outs.size(-1))

        logits = outs.view(outs.size(0), -1, outs.size(-1))

        evidence = self.logits2evidence(logits, evidence_type='exp')
        alpha = evidence + 1

        total_loss = 0
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
        numerator = torch.sum(numerator, -1)
        risk_loss = numerator / denominator
        return torch.sum(risk_loss)

    def calculate_metrics(self, batch_out, average_meter_set):
        outs = batch_out['outs']
        input_ids = batch_out['input_ids']
        masked_lm_ids = batch_out['masked_lm_ids']
        info = batch_out['info']

        num_peers = outs.size(0)
        for model_id in range(num_peers):
            masked_lm_probs, uncertainty = self.calculate_output(batch_out)
            average_meter_set = self.calculate_metrics_one_model(masked_lm_probs[model_id], uncertainty[model_id], input_ids, masked_lm_ids, info, model_id, average_meter_set)
        return average_meter_set

    def calculate_metrics_one_model(self, masked_lm_probs, uncertainty, input_ids, masked_lm_ids, info, model_id, average_meter_set):
        masked_lm_probs = masked_lm_probs.view((-1, self.max_predictions_per_seq, masked_lm_probs.shape[1]))
        uncertainty = uncertainty.view((-1, self.max_predictions_per_seq))

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
            u = uncertainty[idx][0].item()

            for k in self.metric_ks:
                value = 1 if rank < k else 0
                average_meter_set.update('M%s_hit_%s'%(model_id, k), value)
                average_meter_set.update('M%s_ndcg_%s'%(model_id, k), value / np.log2(rank + 2))
            average_meter_set.update('M%s_ap'%model_id, 1.0 / (rank + 1))
            average_meter_set.update('M%s_uap' % model_id, (1.0 - u) / (rank + 1))

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


    @classmethod
    def code(cls):
        return 'FinetuningEDL'