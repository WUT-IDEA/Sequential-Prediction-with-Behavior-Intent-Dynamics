import json
import os
import pprint as pp
import random
from datetime import date
import numpy as np
import torch
import torch.backends.cudnn as cudnn



def setup_train(args):
    set_up_gpu(args)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)
    fix_random_seed_as(args.model_init_seed)

    pp.pprint({k: v for k, v in vars(args).items() if v is not None}, width=1)
    return export_root

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def create_experiment_export_folder(args):
    experiment_dir, experiment_description, dataset_code = args.experiment_dir, args.experiment_description, args.dataset_code
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    if args.resume_experiment_dir == '':
        experiment_path = get_name_of_experiment_path(experiment_dir, dataset_code, experiment_description)
    else:
        experiment_path = os.path.join(experiment_dir, dataset_code, (experiment_description + "_" + args.resume_experiment_dir))
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    print('Folder created: ' + os.path.abspath(experiment_path))
    return experiment_path

def get_name_of_experiment_path(experiment_dir, dataset_code, experiment_description):
    experiment_path = os.path.join(experiment_dir, dataset_code, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path

def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx

def export_experiments_config_as_json(args, experiment_path):
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=2)

def set_up_gpu(args):
    if args.device == 'cpu':
        return
    if args.num_gpu < 0:
        args.num_gpu = -args.num_gpu
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        memory_gpu = ','.join(map(lambda x: str(x), np.argsort(memory_gpu)[::-1][:args.num_gpu]))
        os.environ['CUDA_VISIBLE_DEVICES'] = memory_gpu
        os.system('rm tmp')
        print("Selected GPU:%s" % memory_gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
