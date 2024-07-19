import os
from abc import ABCMeta, abstractmethod
import torch


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))

class LoggerService(object):
    def __init__(self, train_loggers=None, test_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.test_loggers = test_loggers if test_loggers else []

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_test(self, log_data):
        for logger in self.test_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def code(cls):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = {}
            state_dict['state_dict'] = kwargs['state_dict']
            state_dict['epoch'] = epoch
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    @classmethod
    def code(cls):
        return 'RecentModel'

class BestModelLogger(AbstractBaseLogger):
    def __init__(self, best_metric, checkpoint_path, filename='checkpoint-best.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.filename = filename

        self.best_epoch = 0
        self.best_metric_name = best_metric
        self.best_metric_avg = 0

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']
        metric = 0
        num_metric = 0.
        for key in kwargs.keys():
            if self.best_metric_name == key[-len(self.best_metric_name):]:
                metric += kwargs[key]
                num_metric += 1
        metric = metric/num_metric
        if metric >= self.best_metric_avg:
            print("update best model: epoch %s"%epoch)
            self.best_epoch = epoch
            self.best_metric_avg = metric
            state_dict = {}
            state_dict['state_dict'] = kwargs['state_dict']
            state_dict['epoch'] = epoch
            state_dict[self.best_metric_name] = metric
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    @classmethod
    def code(cls):
        return 'BestModel'


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_label='loss', group_name='train'):
        self.key = key
        self.graph_label = graph_label
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['epoch'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['epoch'])

    def complete(self, *args, **kwargs):
        self.writer.close()

    @classmethod
    def code(cls):
        return 'GraphPrinter'
