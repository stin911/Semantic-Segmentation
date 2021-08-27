import torch
from abc import ABC, abstractmethod
from SemanticSegmentation.engine.setting_app import *
from torch.utils.tensorboard import SummaryWriter


class Training(ABC):

    def __init__(self, args):
        if args['tensorboard']:
            self.writer = SummaryWriter()
        self.network = select_model(args)
        self.criterion = select_loss(args['loss'])
        self.optimizer = select_optim(args['optimizer'], lr=args['learning_rate'], net=self.network.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def training(self, epoch, data, val):
        print('train')

    def inference(self, data):
        print('infer')
