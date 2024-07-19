# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import custom_optimization
import tensorflow as tf
import numpy as np
import sys
import pickle
import time
import random

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_integer("SEED", 12345, "SEED.")
flags.DEFINE_float("alpha", 0.5, "The weight of loss.")
flags.DEFINE_float("beta", 0.02, "The truncation.")
flags.DEFINE_float("gamma", 2, "The weight of knowledge.")
flags.DEFINE_integer("peers", 2, "The number of peers.")
flags.DEFINE_string("manner", "min", "learning manner")
flags.DEFINE_string("initializer", "truncated_normal_initializer", "initializer_name")



flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("user_embedding", False, "Whether to add user_embedding.")

#flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

#flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", False, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", None, "vocab filename")
flags.DEFINE_string("user_history_filename", None, "user history filename")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,3' #同时使用GPU 0,1,2

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
# memory_gpu = ','.join(map(lambda x:str(x), np.argsort(memory_gpu)[::-1][0]))
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argsort(memory_gpu)[::-1][0])
os.system('rm tmp')

SEED=FLAGS.SEED
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

class EvalHooks(tf.train.SessionRunHook):
    def __init__(self):
        tf.logging.info('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = [0.0]*FLAGS.peers
        self.hit_1 = [0.0]*FLAGS.peers
        self.ndcg_5 = [0.0]*FLAGS.peers
        self.hit_5 = [0.0]*FLAGS.peers
        self.ndcg_10 = [0.0]*FLAGS.peers
        self.hit_10 = [0.0]*FLAGS.peers
        self.ap = [0.0]*FLAGS.peers


        self.vocab = None

        if FLAGS.user_history_filename is not None:
            print('load user history from :' + FLAGS.user_history_filename)
            with open(FLAGS.user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if FLAGS.vocab_filename is not None:
            print('load vocab from :' + FLAGS.vocab_filename)
            with open(FLAGS.vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        for model_id in range(FLAGS.peers):
            print(
                "\nModel_{}: ndcg@1:{}, hit@1:{}, ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
                format(model_id, self.ndcg_1[model_id] / self.valid_user, self.hit_1[model_id] / self.valid_user,
                       self.ndcg_5[model_id] / self.valid_user, self.hit_5[model_id] / self.valid_user,
                       self.ndcg_10[model_id] / self.valid_user,
                       self.hit_10[model_id] / self.valid_user, self.ap[model_id] / self.valid_user,
                       self.valid_user))

    def before_run(self, run_context):
        #tf.logging.info('run before run')
        #print('run before_run')
        variables = tf.get_collection('eval_sp')
        return tf.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        #tf.logging.info('run after run')
        #print('run after run')
        probs_1, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_probs = [probs_x.reshape(
            (-1, FLAGS.max_predictions_per_seq, probs_x.shape[1])) for probs_x in probs_1]
#         print("loss value:", masked_lm_log_probs.shape, input_ids.shape,
#               masked_lm_ids.shape, info.shape)

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_probs_elem = [masked_lm_probs_x[idx, 0] for masked_lm_probs_x in masked_lm_probs]
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            if FLAGS.use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 100:
                        sampled_ids = np.random.choice(self.ids, 100, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:100]
            else:
                # print("evaluation random -> ")
                for _ in range(99):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = [-masked_lm_probs_elem_x[item_idx] for masked_lm_probs_elem_x in masked_lm_probs_elem]
            rank = [predictions_x.argsort().argsort()[0] for predictions_x in predictions]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            for model_id in range(FLAGS.peers):
                if rank[model_id] < 1:
                    self.ndcg_1[model_id] += 1
                    self.hit_1[model_id] += 1
                if rank[model_id] < 5:
                    self.ndcg_5[model_id] += 1 / np.log2(rank[model_id] + 2)
                    self.hit_5[model_id] += 1
                if rank[model_id] < 10:
                    self.ndcg_10[model_id] += 1 / np.log2(rank[model_id] + 2)
                    self.hit_10[model_id] += 1

                self.ap[model_id] += 1.0 / (rank[model_id] + 1)
            

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""


        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name,
                                                         features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        embedding = modeling.Embedding(config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            info = info,
            use_one_hot_embeddings=use_one_hot_embeddings)

        model_list = []
        for model_id in range(FLAGS.peers):
            model = modeling.BertModel(
                config=bert_config,
                input_ids=input_ids,
                is_training=is_training,
                input_tensor=embedding.get_embedding_output(),
                input_mask=input_mask,
                scope='model_%s'%model_id)
            model_list.append(model)

        input_tensor_list = tf.stack([model.get_sequence_output() for model in model_list], 0)

        (loss_hard, loss_soft, probs_1) = get_masked_lm_output(
            bert_config, input_tensor_list,
            embedding.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op, loss = optimization.create_optimizer(loss_hard, loss_soft,
                                                           FLAGS.alpha,
                                                           learning_rate,
                                                           num_train_steps,
                                                           num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss_hard, probs_1, masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                loss = [tf.metrics.mean(values=loss) for loss in loss_hard]
                masked_lm_probs = [tf.reshape(probs, [-1, probs.shape[-1]]) for probs in probs_1]
                masked_lm_predictions = [tf.argmax(x, axis=-1, output_type=tf.int32) for x in masked_lm_probs]
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = [tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=x,
                    weights=masked_lm_weights) for x in masked_lm_predictions]

                res = {}

                for i,los in enumerate(loss):
                    res['model_%s_masked_lm_accuracy'%i] = masked_lm_accuracy[i]
                    res['model_%s_lm_loss'%i] = los
                return res

            tf.add_to_collection('eval_sp', probs_1)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)

            eval_metrics = metric_fn(loss_hard, probs_1, masked_lm_ids, masked_lm_weights)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=sum(loss_hard),
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec
    return model_fn


def get_masked_lm_output(bert_config, input_tensor_list, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [model_num*batch_size*label_size, dim]
    sequence_shape = modeling.get_shape_list(input_tensor_list, expected_rank=4)
    peers_num = sequence_shape[0]
    gather_list = []
    for i in range(peers_num):
        gather_list.append(gather_indexes(input_tensor_list[i], positions))
    input_tensor = tf.concat(gather_list, 0)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        logits = tf.reshape(logits, [peers_num, -1, output_weights.shape[0]])

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        loss_hard, loss_soft, probs_1 = \
            cal_total_loss(logits, label_weights, one_hot_labels, peers_num)

    return (loss_hard, loss_soft, probs_1)

def cal_total_loss(logits, label_weights, one_hot_labels, peers_num):
    probs_1 = [tf.nn.softmax(logits[i], -1) for i in range(peers_num)]
    log_probs_1 = [tf.nn.log_softmax(logits[i], -1) for i in range(peers_num)]
    log_probs_0 = [tf.log(1 - probs_1[i]) for i in range(peers_num)]
    probs_1_stop_gradient = [tf.stop_gradient(probs_1[i], name="probs_%s1_stop_gradient"%i) for i in range(peers_num)]
    denominator = tf.reduce_sum(label_weights) + 1e-5

    loss_hard, top_order = cal_hard_loss(log_probs_1, one_hot_labels, label_weights, denominator)
    loss_soft = cal_soft_loss(probs_1_stop_gradient, one_hot_labels, log_probs_0, log_probs_1, label_weights,
                                denominator, top_order)
    return loss_hard, loss_soft, probs_1

#tag_prob, one_hot_labels, src_log_probs_0, src_log_probs_1, label_weights, denominator, top_order
def cal_soft_loss1(probs_1_stop_gradient, one_hot_labels, log_probs_0, log_probs_1, label_weights, denominator, top_order):
    gamma = FLAGS.gamma
    loss_soft = []
    peers_num = len(log_probs_1)
    for id in range(peers_num):
        tag_prob_list = probs_1_stop_gradient[:id] + probs_1_stop_gradient[id+1:]
        loss_all_peers = 0
        for tag_prob in tag_prob_list:
            loss = one_hot_labels * tf.pow(1 - tag_prob, gamma) * log_probs_1[id] + (1 - one_hot_labels) * tf.pow(tag_prob, gamma) * log_probs_0[id]
            loss = top_order*label_weights * (-tf.reduce_sum(loss, axis=[-1]))
            numerator = tf.reduce_sum(loss)
            loss = numerator / denominator
            loss_all_peers += loss
        loss_soft.append(loss_all_peers/(peers_num-1))
    return loss_soft

def cal_soft_loss(probs_1_stop_gradient, one_hot_labels, log_probs_0, log_probs_1, label_weights, denominator, top_order):
    gamma = FLAGS.gamma
    loss_soft = []
    peers_num = len(log_probs_1)
    for id in range(peers_num):
        tag_prob_list = probs_1_stop_gradient[:id] + probs_1_stop_gradient[id+1:]
        tag_prob_list = tf.stack(tag_prob_list, 0)

        if FLAGS.manner == "max":
            tmp = one_hot_labels * tag_prob_list
            tmp = tf.reduce_sum(tmp, axis=[-1])
            index = tf.math.argmax(tmp, axis=0)
            batch = tf.cast(tf.shape(index)[0], dtype=index.dtype)
            index = tf.stack([index, tf.range(batch)], axis=-1)
            tag_prob = tf.gather_nd(tag_prob_list, index)
        elif FLAGS.manner == "min":
            tmp = one_hot_labels * tag_prob_list
            tmp = tf.reduce_sum(tmp, axis=[-1])
            index = tf.math.argmin(tmp, axis=0)
            batch = tf.cast(tf.shape(index)[0], dtype=index.dtype)
            index = tf.stack([index, tf.range(batch)], axis=-1)
            tag_prob = tf.gather_nd(tag_prob_list, index)
        elif FLAGS.manner == "mean":
            tag_prob = tf.reduce_mean(tag_prob_list, 0)

        loss = one_hot_labels * tf.pow(1 - tag_prob, gamma) * log_probs_1[id] + (1 - one_hot_labels) * tf.pow(tag_prob, gamma) * log_probs_0[id]
        # loss = label_weights * (-tf.reduce_sum(loss, axis=[-1]))
        loss = top_order*label_weights * (-tf.reduce_sum(loss, axis=[-1]))
        numerator = tf.reduce_sum(loss)
        loss = numerator / denominator
        loss_soft.append(loss)
    return loss_soft

def cal_hard_loss(src_log_probs_list, one_hot_labels, label_weights, denominator):
    loss_list = []
    for src_log_probs in src_log_probs_list:
        loss_tmp = one_hot_labels * src_log_probs
        loss_tmp = label_weights * (-tf.reduce_sum(loss_tmp, axis=[-1]))
        loss_list.append(loss_tmp)

    beta = FLAGS.beta
    shape_number = tf.shape(loss_list[0])[0]
    top_n = tf.cast(tf.cast(shape_number, tf.float32) * (1 - beta), tf.int32)
    top_order = None
    flag = False
    for loss in loss_list:
        order = tf.nn.top_k(-tf.nn.top_k(-loss,shape_number)[1],shape_number)[1]
        top_order = tf.logical_and(order > top_n, top_order) if flag else (order > top_n)
        flag = True

    top_order = tf.cast(tf.logical_not(top_order), tf.float32)
    loss_hard = []
    for loss in loss_list:
        numerator = tf.reduce_sum(loss)
        #numerator = tf.reduce_sum(top_order * loss)
        loss_hard.append(numerator / denominator)
    return loss_hard, top_order


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
            tf.FixedLenFeature([1], tf.int64),  #[user]
            "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            #cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            #d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            #d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)


        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    session_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 1
    session_config.gpu_options.allow_growth = True
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    session = tf.Session(config=session_config)

    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS.checkpointDir = FLAGS.checkpointDir + FLAGS.signature
    print('checkpointDir:', FLAGS.checkpointDir)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    train_input_files = []
    for input_pattern in FLAGS.train_input_file.split(","):
        train_input_files.extend(tf.gfile.Glob(input_pattern))

    test_input_files = []
    if FLAGS.test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in FLAGS.test_input_file.split(","):
            test_input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** train Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tf.logging.info("*** test Input Files ***")
    for input_file in train_input_files:
        tf.logging.info("  %s" % input_file)

    tpu_cluster_resolver = None

    #is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        model_dir=FLAGS.checkpointDir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    
    if FLAGS.vocab_filename is not None:
        with open(FLAGS.vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": FLAGS.batch_size
        })

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(
            input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        #tf.logging.info('special eval ops:', special_eval_ops)
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks()])

        output_eval_file = os.path.join(FLAGS.checkpointDir,
                                        "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            tf.logging.info(bert_config.to_json_string())
            writer.write(bert_config.to_json_string()+'\n')
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("checkpointDir")
    flags.mark_flag_as_required("user_history_filename")
    tf.app.run()
