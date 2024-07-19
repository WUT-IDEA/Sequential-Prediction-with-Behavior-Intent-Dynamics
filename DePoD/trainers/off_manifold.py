#!/usr/bin/env python
# coding:utf-8
"""
Name : off_manifold.py
Author  : issac
Time    : 2022/2/28 22:11
"""
import torch

class off_manifold_samples(object):
    def __init__(self, eps=0.001, rand_init='n', eps_min = 0.001, eps_max=0.1):
        super(off_manifold_samples, self).__init__()
        self.eps = eps
        self.rand_init = rand_init
        self.eps_min = eps_min
        self.eps_max = eps_max

    def generate(self, model, batch_input, ):
        input_ids = batch_input['input_ids']
        input_mask = batch_input
        onehot = batch_input#F.one_hot(masked_lm_ids, outs.size(-1))

        model.eval()
        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                embedding = model.module.get_input_embeddings()(input_ids)
            else:
                embedding = model.get_input_embeddings()(input_ids)

        input_embedding = embedding.detach()
        # random init the adv samples
        if self.rand_init == 'y':
            input_embedding = input_embedding + torch.zeros_like(input_embedding).uniform_(-self.eps, self.eps)

        input_embedding.requires_grad = True

        # input_embedding.grad.detach_()
        # input_embedding.grad.data.zero_()

        if input_embedding.grad is not None:
            input_embedding.grad.data.fill_(0)

        alpha = model(batch_input, input_embedding)[0]
        s = alpha.sum(1, keepdim=True)
        p = alpha / s
        cost = torch.sum((onehot - p) ** 2, dim=1).mean() + \
                   torch.sum(p * (1 - p) / (s + 1), axis=1).mean()
        if torch.cuda.device_count() > 1:
            cost = cost.mean()
        model.zero_grad()
        cost.backward()

        off_samples = input_embedding + self.eps * torch.sign(input_embedding.grad.data)
        off_samples = torch.min(torch.max(off_samples, embedding - self.eps), embedding + self.eps)

        model.train()
        return off_samples.detach()

