import torch
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_bounding_boxes
import monai.networks.nets as nets
from CarLoader import *
from torch.utils.data import DataLoader
from torchsummary import summary
from transfor import *





class model:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nets.basicunet(dimensions=args["dimensions"], in_channels=args["in_channels"]).to(self.device)


class Training_seg:
    def __init__(self, batch, num_work, args):
        self.net = model(args=args)
        self.batch_n = batch
        self.num_work = num_work
        self.train_loader = None
        self.val_loader = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.net.parameters(), lr=0.0001, momentum=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_data(self, t_image_path, t_mask_path, v_image_path=None, v_mask_path=None):
        data = DataCarData(t_image_path, t_mask_path, transform=transforms.Compose([ResizeMine(), ToTensor()]))

        self.train_loader = DataLoader(data, batch_size=self.batch_n, num_workers=self.num_work)
        """data = DataCarData(v_image_path, v_mask_path)
        self.val_loader = DataLoader(data, batch_size=self.batch_n, num_workers=self.num_work)"""

    def start(self):
        for i, sample in enumerate(self.train_loader):
            out = self.net.net(sample['image'].to(self.device))
            print(i)
            print(out)

    def summary(self):
        """
        :return: generate a summary of the model
        """
        summary(self.net.net, (3, 512, 512))


def main():
    sett = {'dimensions': 2, 'in_channels': 3}
    training = Training_seg(2, num_work=1, args=sett)
    training.initialize_data("D:/Car-dataset/train/", "D:/Car-dataset/Mask_train/")
    #training.start()
    print(training.summary())


if __name__ == '__main__':
    main()
