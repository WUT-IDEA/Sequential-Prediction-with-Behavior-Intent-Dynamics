from trainers.loggers import *
from utils import AverageMeterSet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pprint import pprint
import json
from abc import *
from pathlib import Path
from trainers.optim import ScheduledOptim


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, test_loader, export_root):
        self.args = args
        self.device = torch.device("%s"%args.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.peer_num = len(args.base_models_name.split(','))
        self.is_parallel = args.num_gpu > 1
        self.load_pretrain_path = False
        self.base_models_num = len(args.base_models_name.split(','))
        if args.load_pretrain:
            self.load_pretrain_path = os.path.join(args.experiment_dir, args.dataset_code, args.load_pretrain)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks

        self.export_root = export_root
        self.best_metric = args.best_metric
        self.log_test = args.log_test
        self.writer, train_loggers, test_loggers = self._create_loggers()
        self.logger_service = LoggerService(train_loggers, test_loggers)

        self.param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'layer_norm', 'beta']

        # shared_para = ['embedding', 'transform',' predicion']
        # if self.peer_num > 2:
        #     optimizer_grouped_parameters = [
        #         {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay+shared_para)],
        #          'weight_decay_rate': 1e-2},
        #         {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in shared_para)],
        #          'weight_decay_rate': 0.0},
        #         {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in shared_para) and not any(nd in n for nd in no_decay)],
        #          'weight_decay_rate': 1e-2},
        #         {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay) and any(nd in n for nd in shared_para)],
        #          'weight_decay_rate': 0.0}
        #     ]
        # else:
        #     optimizer_grouped_parameters = [
        #         {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],
        #          'weight_decay_rate': 1e-2},
        #         {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)],
        #          'weight_decay_rate': 0.0}
        #     ]

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 1e-2},
            {'params': [p for n, p in self.param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        total_iterations = self.num_epochs * len(self.train_loader)
        self.optimizer = ScheduledOptim(
            optim.AdamW(optimizer_grouped_parameters,
                       betas=(0.9, 0.999), eps=1e-06), self.peer_num, args.n_warmup_steps, args.init_lr, total_iterations)

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self):
        pass

    @abstractmethod
    def calculate_metrics(self):
        pass

    def resume(self):
        recent_model_path = os.path.join(self.export_root, 'models', 'checkpoint-recent.pth')
        if os.path.exists(recent_model_path):
            checkpoint = torch.load(recent_model_path)
            self.model.load_state_dict(checkpoint['state_dict']['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['state_dict']['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            return start_epoch
        elif self.load_pretrain_path:
            load_pretrain_path = os.path.join(self.load_pretrain_path, 'models', 'checkpoint-best.pth')
            checkpoint = torch.load(load_pretrain_path)
            self.model.load_state_dict(checkpoint['state_dict']['model_state_dict'])
            return 0
        else:
            return 0

    def train(self):
        start_epoch = self.resume()

        if self.is_parallel:
            self.model = nn.DataParallel(self.model)

        for epoch in range(start_epoch, self.num_epochs):
            self.train_one_epoch(epoch)
        self.writer.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        average_meter_set = AverageMeterSet()

        tqdm_dataloader = tqdm(self.train_loader)

        for batch_idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

            batch_input = {
                'info': info,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_ids': masked_lm_ids,
                'masked_lm_weights': masked_lm_weights
            }

            batch_out = self.model(batch_input)

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch_out, current_epoch=epoch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step_and_update_lr()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                'Epoch {}, loss {:.3f} '.format(epoch+1, average_meter_set['loss'].avg))

        tqdm_dataloader.set_description('Logging to Tensorboard')
        log_data = {
            'state_dict': (self._create_state_dict()),
            'epoch': epoch+1
        }

        if self.log_test:
            self.record_test_result(log_data, epoch)

        log_data.update(average_meter_set.averages())
        self.logger_service.log_train(log_data)     #save model and loss after one epoch


    def record_test_result(self, log_data, epoch):
        self.model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

                batch_input = {
                    'info': info,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'masked_lm_positions': masked_lm_positions,
                    'masked_lm_ids': masked_lm_ids,
                    'masked_lm_weights': masked_lm_weights
                }

                batch_out = self.model(batch_input)
                loss = self.calculate_loss(batch_out, current_epoch=epoch)
                average_meter_set.update('loss', loss.item())
                average_meter_set = self.calculate_metrics(batch_out, average_meter_set)

                tqdm_dataloader.set_description(
                    'Test: {} {:.4f} '.format(self.best_metric, average_meter_set['M0_%s'%self.best_metric].avg))

            average_metrics = average_meter_set.averages()
            log_data.update(average_metrics)

            self.logger_service.log_test(log_data)


    def test(self, model):
        self.model = model.to(self.device)
        print('\nTest model with test set!')
        test_model_path = os.path.join(self.export_root, 'models', 'checkpoint-recent.pth')
        checkpoint = torch.load(test_model_path)
        self.model.load_state_dict(checkpoint['state_dict']['model_state_dict'])

        self.model.eval()
        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                info, input_ids, input_mask, masked_lm_positions, masked_lm_ids, masked_lm_weights = batch

                batch_input = {
                    'info': info,
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'masked_lm_positions': masked_lm_positions,
                    'masked_lm_ids': masked_lm_ids,
                    'masked_lm_weights': masked_lm_weights
                }

                batch_out = self.model(batch_input)

                average_meter_set = self.calculate_metrics(batch_out, average_meter_set)

            average_metrics = average_meter_set.averages()
            count_metrics = average_meter_set.counts()
            average_metrics['user_num'] = count_metrics['M0_hit_1']
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            pprint(average_metrics)

    def _create_loggers(self):
        root = Path(self.export_root)
        writer = SummaryWriter(root.joinpath('logs'))
        model_checkpoint = root.joinpath('models')

        train_loggers = [
            MetricGraphPrinter(writer, key='loss', graph_label='Loss', group_name='Train'),
            RecentModelLogger(model_checkpoint, filename='checkpoint-recent.pth')
        ]

        test_loggers = [
            MetricGraphPrinter(writer, key='loss', graph_label='Loss', group_name='Test'),
            BestModelLogger(self.best_metric, model_checkpoint, filename='checkpoint-best.pth')
        ]


        for model_id in range(self.peer_num):
            for k in self.metric_ks:
                test_loggers.append(
                    MetricGraphPrinter(writer, key='M%s_hit_%s'%(model_id, k), graph_label='M%s_hit_%s'%(model_id, k), group_name='Test'))
                test_loggers.append(
                    MetricGraphPrinter(writer, key='M%s_ndcg_%s'%(model_id, k), graph_label='M%s_ndcg_%s'%(model_id, k), group_name='Test'))
            test_loggers.append(
                MetricGraphPrinter(writer, key='M%s_ap'%model_id, graph_label='M%s_ap'%model_id, group_name='Test'))
        return writer, train_loggers, test_loggers



    def _create_state_dict(self):
        return {
            'model_state_dict': self.model.module.state_dict() if self.is_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
