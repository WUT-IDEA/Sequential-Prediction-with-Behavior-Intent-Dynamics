#!/usr/bin/env python
# coding:utf-8
"""
Name : instance.py
Author  : issac
Time    : 2021/10/18 21:43
"""
import torch
import collections
from torch.utils.data import Dataset

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels, corrupt_tokens=None):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.corrupt_tokens = corrupt_tokens if corrupt_tokens is not None else None

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([str(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([str(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


class TrainingDataset(Dataset):
    def __init__(self, instances, vocab, args):
        self.device = torch.device("%s" % 'cuda' if torch.cuda.is_available() else "cpu")
        self.dataset = self._creat_dataset(instances, args.max_seq_length, args.max_predictions_per_seq, vocab)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        info = self.dataset[index]["info"]
        input_ids = self.dataset[index]["input_ids"]
        input_mask = self.dataset[index]["input_mask"]
        masked_lm_positions = self.dataset[index]["masked_lm_positions"]
        masked_lm_ids = self.dataset[index]["masked_lm_ids"]
        masked_lm_weights = self.dataset[index]["masked_lm_weights"]
        if self.dataset[index]["input_corrupt_ids"] is not None:
            input_corrupt_ids = self.dataset[index]["input_corrupt_ids"]
            return info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights, input_corrupt_ids
        return info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights

    def print_index(self, index):
        info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights = self.__getitem__(index)
        print("*********print the %s th sample in dataset.**********"%index)
        print("info:", info)
        print("input_ids:", input_ids)
        print("input_mask:", input_mask)
        print("masked_lm_positions:", masked_lm_positions)
        print("masked_lm_ids:", masked_lm_ids)
        print("masked_lm_weights:", masked_lm_weights)

    def _creat_dataset(self, instances, max_seq_length, max_predictions_per_seq, vocab):
        dataset = []
        for (inst_index, instance) in enumerate(instances):
            try:
                input_ids = vocab.convert_tokens_to_ids(instance.tokens)
                if instance.corrupt_tokens is not None:
                    input_corrupt_ids = vocab.convert_tokens_to_ids(instance.corrupt_tokens)
                    input_corrupt_ids += [0] * (max_seq_length - len(input_corrupt_ids))
            except:
                print(instance)

            input_mask = [1] * len(input_ids)
            assert len(input_ids) <= max_seq_length

            input_ids += [0] * (max_seq_length - len(input_ids))
            input_mask += [0] * (max_seq_length - len(input_mask))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
            masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
            masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

            features = collections.OrderedDict()
            features["info"] = torch.tensor(instance.info, device=self.device)
            features["input_ids"] = torch.tensor(input_ids, device=self.device)
            features["input_mask"] = torch.tensor(input_mask, device=self.device)
            features["masked_lm_positions"] = torch.tensor(masked_lm_positions, device=self.device)
            features["masked_lm_ids"] = torch.tensor(masked_lm_ids, device=self.device)
            features["masked_lm_weights"] = torch.tensor(masked_lm_weights, device=self.device)
            features['input_corrupt_ids'] = torch.tensor(input_corrupt_ids, device=self.device) if instance.corrupt_tokens else None
            dataset.append(features)
        return dataset