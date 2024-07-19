#!/usr/bin/env python
# coding:utf-8

from .trainer3 import Trainer       #### 测试修改了trainer3
from .trainerEDL import TrainerEDL
from .FinetuningEDL import FinetuningEDL

TRAINERS = {
    Trainer.code(): Trainer,
    TrainerEDL.code(): TrainerEDL,
    FinetuningEDL.code(): FinetuningEDL
}

def trainer_factory(args, model, train_dataloader, test_dataloader, export_root):
    trainer = TRAINERS[args.trainer_code]
    trainer = trainer(args, model, train_dataloader, test_dataloader, export_root)
    return trainer
