  #!/usr/bin/env python
# coding:utf-8
from collections import defaultdict
from pathlib import Path
import numpy as np
import pickle
import multiprocessing
import time
from dataloader.vocab import FreqVocab
import random
from dataloader.instance import TrainingInstance, TrainingDataset
import collections
from pprint import pprint
from torch.utils.data import DataLoader
import torch

class Dataloader(object):
    def __init__(self, args):
        self.dataset_code = args.dataset_code
        self.args = args
        self.dataset_dir = Path().joinpath('datasets', self.dataset_code)
        self.device = torch.device("%s" % 'cuda' if torch.cuda.is_available() else "cpu")
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir()

    def load_dataset(self):
        args = self.args
        if not self.dataset_dir.joinpath('%s' % args.version_id).exists():
            self.dataset_dir.joinpath('%s' % args.version_id).mkdir()
        output_train_file = self.dataset_dir.joinpath('%s' % args.version_id, '%s.train.pkl' % args.version_id)
        output_test_file = self.dataset_dir.joinpath('%s' % args.version_id, '%s.test.pkl' % args.version_id)
        vocab_file = self.dataset_dir.joinpath('vocab.pkl')
        history_file = self.dataset_dir.joinpath('history.pkl')
        if output_train_file.exists() and output_test_file.exists():
            print('The dataset has already been preprocessed.')
        else:
            train, val, test, statistics = self._data_partition()

            # save and print statistics
            statistics_file = self.dataset_dir.joinpath('statistics.pkl')
            with statistics_file.open('wb') as f:
                pickle.dump(statistics, f)
            pprint(statistics)

            # put validate into train
            for u in train:
                if u in val:
                    train[u].extend(val[u])

            # get the max index of the data
            user_train_data = {
                'user_' + str(k): ['item_' + str(item) for item in v]
                for k, v in train.items() if len(v) > 0
            }
            user_test_data = {
                'user_' + str(u):
                    ['item_' + str(item) for item in (train[u] + test[u])]
                for u in train if len(train[u]) > 0 and len(test[u]) > 0
                }

            vocab = FreqVocab(user_test_data)
            user_test_data_output = {
                k: [vocab.convert_tokens_to_ids(v)]
                for k, v in user_test_data.items()
            }

            with vocab_file.open('wb') as f:
                pickle.dump(vocab, f)
            with history_file.open('wb') as f:
                pickle.dump(user_test_data_output, f)

            print('begin to generate training samples')
            self._gen_samples(output_train_file, args, user_train_data, vocab, force_last=False)

            print('begin to generate test samples')
            self._gen_samples(output_test_file, args, user_test_data, vocab, force_last=True)

        with output_train_file.open('rb') as f:
            train_dataset = pickle.load(f)
        with output_test_file.open('rb') as f:
            test_dataset = pickle.load(f)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
        return train_dataloader, test_dataloader

    def _gen_samples(self, output_filename, args, all_documents_raw, vocab, force_last=False):
        rng = random.Random(args.random_shuffle_seed)
        instances = self._create_training_instances(all_documents_raw,
                                       args.max_seq_length,
                                       args.dupe_factor,
                                       args.masked_lm_prob,
                                       args.max_predictions_per_seq,
                                       rng,
                                       vocab,
                                       args.mask_prob,
                                       args.prop_sliding_window,
                                       args.pool_size,
                                       force_last)
        dataset = TrainingDataset(instances, vocab, args)
        with output_filename.open('wb') as f:
            pickle.dump(dataset, f)

    def _create_training_instances(self,
                                  all_documents_raw,
                                  max_seq_length,
                                  dupe_factor,
                                  masked_lm_prob,
                                  max_predictions_per_seq,
                                  rng,
                                  vocab,
                                  mask_prob,
                                  prop_sliding_window,
                                  pool_size,
                                  force_last=False):
        """Create `TrainingInstance`s from raw text."""
        all_documents = {}

        if force_last:
            max_num_tokens = max_seq_length
            for user, item_seq in all_documents_raw.items():
                if len(item_seq) == 0:
                    print("got empty seq:" + user)
                    continue
                all_documents[user] = [item_seq[-max_num_tokens:]]
        else:
            max_num_tokens = max_seq_length  # we need two sentence

            sliding_step = (int)(
                prop_sliding_window *
                max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
            for user, item_seq in all_documents_raw.items():
                if len(item_seq) == 0:
                    print("got empty seq:" + user)
                    continue

                # todo: add slide
                if len(item_seq) <= max_num_tokens:
                    all_documents[user] = [item_seq]
                else:
                    beg_idx = list(range(len(item_seq) - max_num_tokens, 0, -sliding_step))
                    beg_idx.append(0)
                    all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

        instances = []
        if force_last:
            for user in all_documents:
                instances.extend(
                    self._create_instances_from_document_test(
                        all_documents, user, max_seq_length))
            print("num of instance:{}".format(len(instances)))
        else:
            start_time = time.perf_counter()
            pool = multiprocessing.Pool(processes=pool_size)
            instances = []
            print("document num: {}".format(len(all_documents)))

            def log_result(result):
                print("callback function result type: {}, size: {} ".format(type(result), len(result)))
                instances.extend(result)

            for step in range(dupe_factor):
                pool.apply_async(
                    self._create_instances_threading, args=(
                        all_documents, max_seq_length,
                        masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
                        mask_prob, step), callback=log_result)
            pool.close()
            pool.join()

            for user in all_documents:
                instances.extend(
                    self._mask_last(all_documents, user, max_seq_length))

            print("num of instance:{}; time:{}".format(len(instances), time.perf_counter() - start_time))
        rng.shuffle(instances)
        return instances

    def _create_instances_threading(self, all_documents, max_seq_length, masked_lm_prob,
                                   max_predictions_per_seq, vocab, rng, mask_prob, step):
        cnt = 0;
        start_time = time.perf_counter()
        instances = []
        for user in all_documents:
            cnt += 1;
            if cnt % 1000 == 0:
                print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt,
                                                                      time.perf_counter() - start_time))
                start_time = time.perf_counter()
            instances.extend(self._create_instances_from_document_train(
                all_documents, user, max_seq_length,
                masked_lm_prob, max_predictions_per_seq, vocab, rng,
                mask_prob))

        return instances

    def _mask_last(self, all_documents, user, max_seq_length):
        """Creates `TrainingInstance`s for a single document."""
        document = all_documents[user]
        max_num_tokens = max_seq_length

        instances = []
        info = [int(user.split("_")[1])]

        for tokens in document:
            assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

            (tokens, masked_lm_positions,
             masked_lm_labels) = self._create_masked_lm_predictions_force_last(tokens)
            instance = TrainingInstance(
                info=info,
                tokens=tokens,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        return instances

    def _create_instances_from_document_train(self,
            all_documents, user, max_seq_length, masked_lm_prob,
            max_predictions_per_seq, vocab, rng, mask_prob):
        """Creates `TrainingInstance`s for a single document."""
        document = all_documents[user]

        max_num_tokens = max_seq_length

        instances = []
        info = [int(user.split("_")[1])]
        vocab_items = vocab.get_items()

        for tokens in document:
            assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

            (tokens, masked_lm_positions,
             masked_lm_labels) = self._create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq,
                vocab_items, rng, mask_prob)
            instance = TrainingInstance(
                info=info,
                tokens=tokens,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)

        return instances

    def _create_instances_from_document_test(self, all_documents, user, max_seq_length):
        """Creates `TrainingInstance`s for a single document."""
        document = all_documents[user]
        max_num_tokens = max_seq_length

        assert len(document) == 1 and len(document[0]) <= max_num_tokens

        tokens = document[0]
        assert len(tokens) >= 1

        (tokens, masked_lm_positions,
         masked_lm_labels) = self._create_masked_lm_predictions_force_last(tokens)

        info = [int(user.split("_")[1])]
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)

        return [instance]

    def _create_masked_lm_predictions_force_last(self, tokens):
        """Creates the predictions for the masked LM objective."""

        last_index = -1
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
                continue
            last_index = i

        assert last_index > 0

        output_tokens = list(tokens)
        output_tokens[last_index] = "[MASK]"

        masked_lm_positions = [last_index]
        masked_lm_labels = [tokens[last_index]]

        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def _create_masked_lm_predictions(self, tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng,
                                     mask_prob):
        """Creates the predictions for the masked LM objective."""

        MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token not in vocab_words:
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < mask_prob:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                    masked_token = rng.choice(vocab_words)

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def statistics(self):
        statistics_file = self.dataset_dir.joinpath('statistics.pkl')
        if statistics_file.exists():
            with statistics_file.open('rb') as f:
                statistics = pickle.load(f)
                pprint(statistics)
        else:
            print('statistics file is missing!')

    def _data_partition(self):
        fname = Path().joinpath('datasets', self.dataset_code + '.txt')
        users = set()
        items = set()
        User = defaultdict(list)
        user_train = {}
        user_val = {}
        user_test = {}
        # assume user/item index starting from 1
        f = open(fname, 'r')
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            users.add(u)
            items.add(i)
            User[u].append(i)
        total_length = []
        max_len = 0
        min_len = 100000
        for user in User:
            nfeedback = len(User[user])
            total_length.append(nfeedback)
            max_len = max(nfeedback, max_len)
            min_len = min(nfeedback, min_len)
            if nfeedback < 3:
                user_train[user] = User[user]
                user_val[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_val[user] = []
                user_val[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        statistics = {'usernum': len(users),
                      'itemnum': len(items),
                      'length':{
                          'sum': np.sum(total_length),
                          'max': max_len,
                          'min': min_len,
                          'mean': np.mean(total_length),
                          'std': np.std(total_length)
                      }
                      }
        return [user_train, user_val, user_test, statistics]

    @classmethod
    def code(self):
        return 'Dataloader'