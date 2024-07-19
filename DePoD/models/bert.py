#!/usr/bin/env python
# coding:utf-8
"""
Name : base.py
Author  : issac
Time    : 2021/10/27 19:31
"""
import torch.nn as nn
from models.bert_modules.transformer import TransformerBlock

class BERT(nn.Module):
    def __init__(self, hidden_units, bert_num_heads, bert_intermediate_size, bert_dropout, bert_num_blocks):
        super().__init__()

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden_units, bert_num_heads, bert_intermediate_size, bert_dropout) for _ in
             range(bert_num_blocks)])

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x, mask):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x