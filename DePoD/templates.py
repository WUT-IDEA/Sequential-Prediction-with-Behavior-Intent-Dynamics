#!/usr/bin/env python
# coding:utf-8
"""
Name : templates.py
Author  : issac
Time    : 2022/2/28 14:54
"""

dataset_items = {'toys': 11924,
                 'nyc16': 440,
                 'chi18': 246,
                 'beauty': 12101}

def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('mutual'):
        args.trainer_code = 'Trainer'
        args.dataloader_code = 'Dataloader'
        args.dataset_code = 'nyc16'
        args.experiment_description = '%s_test'%args.dataset_code
        args.init_lr  = 1e-3
        args.version_id = 'nip_test2'
        args.max_seq_length = 200
        args.num_epochs = 5
        args.num_items = dataset_items[args.dataset_code]
        args.resume_experiment_dir = ''
        args.device = 'cuda'
        args.base_models_name = 'cnn,gru'
        args.learning_pattern = 'mutual_all'
        args.alpha = 5.0
        args.beta = 1.0
        args.gamma = 1.0
        args.num_negative_elements = 100
        args.non_target_sampling_strategy='pop_first'
        args.batch_size=8
        args.dupe_factor = 0
        args.max_predictions_per_seq = 1

    elif args.template.startswith('case'):
        args.trainer_code = 'Trainer'
        args.dataloader_code = 'Dataloader'
        args.dataset_code = 'nyc16'
        args.experiment_description = '%s_all_m2_new'%args.dataset_code
        args.init_lr  = 1e-3
        args.version_id = 'bertv1'
        args.max_seq_length = 200
        args.num_epochs = 5
        args.num_items = dataset_items[args.dataset_code]
        args.resume_experiment_dir = '2022-09-19_0'
        args.device = 'cuda'
        args.base_models_name = 'bert,bert'
        args.learning_pattern = 'mutual_all'
        args.alpha = 5.0
        args.beta = 1.0
        args.gamma = 1.0
        args.num_negative_elements = 100
        args.non_target_sampling_strategy='pop_first'
        args.batch_size=256

    elif args.template.startswith('separate'):
        args.trainer_code = 'Trainer'
        args.dataloader_code = 'Dataloader'
        args.dataset_code = 'beauty'
        args.experiment_description = '%s_test' % args.dataset_code
        args.init_lr = 1e-3
        args.version_id = 'bertv2'
        args.max_seq_length = 100
        args.num_epochs = 30
        args.num_items = dataset_items[args.dataset_code]
        args.resume_experiment_dir = ''#'2023-07-03_1'
        args.device = 'cuda'
        args.base_models_name = 'bert'
        args.learning_pattern = 'mutual_all'#'separate'
        args.alpha = 5.0
        args.beta = 1.0
        args.gamma = 1.0
        args.num_negative_elements = 100
        args.non_target_sampling_strategy = 'pop_first'
        args.batch_size = 256
        args.random_shuffle_seed = 679


    elif args.template.startswith('standard'):
        args.trainer_code = 'Trainer'
        args.dataloader_code = 'Dataloader'
        args.dataset_code = 'toys'
        args.experiment_description = '%s_Standard'%args.dataset_code
        args.factor = 1
        args.version_id = 'bertv1'
        args.num_epochs = 400
        args.num_items = dataset_items[args.dataset_code]
        args.resume_experiment_dir = '2022-03-14_0'
        args.device = 'cpu'
        args.learning_pattern = 'separate'

    elif args.template.startswith('evidence'):
        args.trainer_code = 'TrainerEDL'
        args.dataloader_code = 'Dataloader'
        args.dataset_code = 'toys'
        args.experiment_description = '%s_test'%args.dataset_code
        args.factor = 1
        args.loss_term = 'ml'
        args.version_id = 'bertv1'
        args.num_epochs = 400
        args.num_items = dataset_items[args.dataset_code]

    elif args.template.startswith('finetuningEDL'):
        args.trainer_code = 'FinetuningEDL'
        args.dataloader_code = 'DataloaderEDL'
        args.dataset_code = 'toys'
        args.experiment_description = 'all_finetuning'#'%s_test_f1'%args.dataset_code
        args.factor = 5e-4
        args.loss_term = 'ml+cl'
        args.load_pretrain = 'toys_Standard_2022-03-14_0'
        args.version_id = 'EDL-finetuningv1'
        args.num_epochs = 200
        args.num_items = dataset_items[args.dataset_code]
        args.device = 'cuda'
        args.n_warmup_steps = 1000
        args.off_mainfold_eps = 0.1
        # args.best_metric = 'uap'
        # args.resume_experiment_dir = '2022-03-07_0'#'2022-03-05_0'#7


