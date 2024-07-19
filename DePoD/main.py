from options import args
# from models.model import Model
from models.model_cnn_gru import Model
from trainers import trainer_factory
from dataloader import dataloader_factory
from utils import *
# from torchsummary import summary

def run():
    export_root = setup_train(args)
    dataloader = dataloader_factory(args)
    train_dataloader, test_dataloader = dataloader.load_dataset()

    model = Model(args)
    # summary(model)
    trainer = trainer_factory(args, model, train_dataloader, test_dataloader, export_root)
    trainer.train()
    trainer.test(model)

    # modellist = []
    # for i in range(5):
    #     modellist.append(Model(args))
    # trainer = trainer_factory(args, modellist, train_dataloader, test_dataloader, export_root)
    # trainer.test(modellist)

if __name__ == '__main__':
    run()

# tensorboard --logdir=logs
