#!/usr/bin/env python
# coding:utf-8

from .dataloader_density import Dataloader
from .dataloaderEDL import DataloaderEDL

DATALOADERS = {
    Dataloader.code(): Dataloader,
    DataloaderEDL.code(): DataloaderEDL
}

def dataloader_factory(args):
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args)
    return dataloader