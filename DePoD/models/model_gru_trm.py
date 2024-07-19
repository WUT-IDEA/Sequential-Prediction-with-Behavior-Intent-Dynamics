#!/usr/bin/env python
# coding:utf-8
import torch.nn as nn
import torch
from models.bert_modules.embedding import TokenEmbedding, PositionalEmbedding
from utils import fix_random_seed_as

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        fix_random_seed_as(args.model_init_seed)

        # parameters
        # embedding
        num_items = args.num_items
        vocab_size = num_items + 2  # mask + padding(0)
        max_position_embeddings = args.max_position_embeddings
        hidden_units = args.hidden_units
        # embedding_dropout = args.embedding_dropout
        # bert model
        bert_num_layers = args.bert_num_layers
        self.bert_num_heads = args.bert_num_heads
        bert_dropout = args.bert_dropout
        bert_intermediate_size = args.bert_intermediate_size
        self.base_models_name = args.base_models_name.split(',')

        self.embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden_units)
        self.position = PositionalEmbedding(max_len=max_position_embeddings, d_model=hidden_units)

        self.base_models = nn.ModuleList()
        for name in self.base_models_name:
            if name == 'bert':
                encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=self.bert_num_heads,
                                                            dim_feedforward=bert_intermediate_size,
                                                            dropout=bert_dropout, activation="gelu", batch_first=True,
                                                            norm_first=True)
                model = nn.TransformerEncoder(encoder_layers, num_layers=bert_num_layers)
            elif name == 'gru':
                model = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, num_layers=bert_num_layers,
                             batch_first=True, dropout=bert_dropout)
            self.base_models.append(model)

        self.transform = nn.Linear(hidden_units, hidden_units)
        self.prediction = nn.Linear(hidden_units, vocab_size, bias=True)
        self.prediction.weight = self.embedding.weight
        self.x_logit_scale = (hidden_units ** -0.5)
        if len(self.base_models_name) > 2:
            self.tea_scorer = nn.Sequential(nn.Linear(hidden_units, hidden_units, bias=True), nn.Tanh(), nn.Linear(hidden_units, 1, bias=True))
            # self.tea_scorer = nn.Linear(vocab_size, 1, bias=True)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            # 对weights进行初始化
            if p.dim() > 1:
                nn.init.trunc_normal_(p, std=0.02)
            # 对bias进行初始化
            else:
                nn.init.constant_(p, 0)

    def forward(self, batch_input, inputs_other=None):
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        input_ids = batch_input['input_ids']
        input_mask = (batch_input['input_mask']==0)
        positions = batch_input['masked_lm_positions']

        # mask = input_mask.unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
        # mask = input_mask.unsqueeze(1).repeat(self.bert_num_heads, input_ids.size(1), 1).float()

        # embedding the indexed sequence to sequence of vectors
        if inputs_other is None:
            gru_embedding = self.embedding(input_ids)
            bert_embedding = gru_embedding + self.position(input_ids)
        elif inputs_other.ndim == 3:
            embedding_out = inputs_other
        elif inputs_other.ndim == 2:
            embedding_out = self.embedding(inputs_other)

        encoder_outs = []
        # seq_embeddings = []
        for i, name in enumerate(self.base_models_name):
            if name == 'bert':
                encoder_out = self.base_models[i](src=bert_embedding, src_key_padding_mask=input_mask)
            elif name == 'gru':
                encoder_out = self.base_models[i](input=gru_embedding)[0]
            # seq_embedding = torch.sum(input_mask.unsqueeze(-1)*out, dim=-2)/torch.sum(input_mask.unsqueeze(-1), dim=-2)
            encoder_out = self._gather_indexes(encoder_out, positions)

            # final_embedding = self.transform(out)
            # seq_embeddings.append(seq_embedding)
            # out = self.prediction(final_embedding) * self.x_logit_scale
            encoder_outs.append(encoder_out)

        encoder_outs = torch.stack(encoder_outs, dim=1)
        model_outs = self.transform(encoder_outs)
        model_outs = self.prediction(model_outs) * self.x_logit_scale

        tea_score = self.tea_scorer(encoder_outs.detach()) if encoder_outs.shape[1] > 2 else None
        # seq_embeddings = torch.stack(seq_embeddings, dim=1)

        batch_out = {
            'outs': model_outs,
            'tea_score': tea_score,
            'masked_lm_ids': batch_input['masked_lm_ids'],
            'masked_lm_weights': batch_input['masked_lm_weights'],
            'input_ids': batch_input['input_ids'],
            'info': batch_input['info'],
            'seq_embeddings': None#seq_embeddings
        }
        return batch_out

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = sequence_tensor.shape
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = torch.reshape(
            torch.arange(0, batch_size) * seq_length, [-1, 1]).to(positions.device)
        flat_positions = torch.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = torch.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = torch.index_select(input=flat_sequence_tensor, dim=0, index=flat_positions)
        return output_tensor

    def get_input_embeddings(self):
        return self.embedding
