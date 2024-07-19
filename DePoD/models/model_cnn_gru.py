#!/usr/bin/env python
# coding:utf-8
import torch.nn as nn
import torch
from models.bert_modules.embedding import TokenEmbedding
from utils import fix_random_seed_as
import torch.nn.functional as F

class Caser(nn.Module):
    def __init__(self,args):
        super(Caser, self).__init__()

        # load parameters info
        self.n_h = 2
        self.n_v = 2
        self.max_seq_length = args.max_seq_length
        self.embedding_size = args.hidden_units
        self.dropout_prob = args.bert_dropout

        # vertical conv layer
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(0, self.max_seq_length, 10)]
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size)) for i in lengths
        ])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        self.fc1 = nn.Linear(fc1_dim_in, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

    def forward(self, inputs):
        # Embedding Look-up
        # use unsqueeze() to get a 4-D input for convolution layers. (batch_size * 1 * max_length * embedding_size)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        inputs = inputs.unsqueeze(1)
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(inputs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(inputs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)
        # fully-connected layer
        seq_output = self.ac_fc(self.fc1(out))
        # the hidden_state of the predicted item, size:(batch_size * hidden_size)
        return seq_output


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        fix_random_seed_as(args.model_init_seed)

        # parameters
        # embedding
        num_items = args.num_items
        vocab_size = num_items + 2  # mask + padding(0)
        hidden_units = args.hidden_units
        # embedding_dropout = args.embedding_dropout
        # bert model
        bert_num_layers = args.bert_num_layers
        self.bert_num_heads = args.bert_num_heads
        bert_dropout = args.bert_dropout
        bert_intermediate_size = args.bert_intermediate_size
        self.base_models_name = args.base_models_name.split(',')


        self.embedding = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden_units)

        self.base_models = nn.ModuleList()
        for name in self.base_models_name:
            encoder = None
            if name == 'cnn':
                encoder = Caser(args)
            elif name == 'gru':
                encoder = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, num_layers=bert_num_layers, batch_first=True,
                     dropout=bert_dropout)
            self.base_models.append(encoder)

        self.transform = nn.Linear(hidden_units, hidden_units)
        self.prediction = nn.Linear(hidden_units, vocab_size, bias=True)
        self.prediction.weight = self.embedding.weight
        self.x_logit_scale = (hidden_units ** -0.5)

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
            embedding_out = self.embedding(input_ids)
        elif inputs_other.ndim == 3:
            embedding_out = inputs_other
        elif inputs_other.ndim == 2:
            embedding_out = self.embedding(inputs_other)

        encoder_outs = []
        # seq_embeddings = []
        for index, base in enumerate(self.base_models):
            if self.base_models_name[index]  == 'cnn':
                encoder_out = base(inputs=embedding_out)
            elif self.base_models_name[index]  == 'gru':
                encoder_out = base(input=embedding_out)[0]
                encoder_out = self._gather_indexes(encoder_out, positions)

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
            'seq_embeddings': None  # seq_embeddings
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
