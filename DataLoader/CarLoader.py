from torch.utils.data import Dataset
import torch
import cv2

import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
from SemanticSegmentation.general_utils.general_transform import *
from torchvision import transforms

from torchvision.transforms import ToTensor


class DataCarData(Dataset):
    """
    Class represent the dataset
    """

    def __init__(self, train_dir, mask_dir, transform=None):
        """
        Args:

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_list = os.listdir(train_dir)
        self.mask_dir = mask_dir
        self.image_dir = train_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: an element of dataset
        """
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image_path).convert("RGB")
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.image_list[idx][:-4] + "_mask.gif")
            mask = Image.open(mask_path)
        else:
            mask = None
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    t = DataCarData("D:/Car-dataset/train/", "D:/Car-dataset/Mask_train/",
                    transform=transforms.Compose([ResizeMine(), CToTensor()]))
    data = DataLoader(t, batch_size=1, num_workers=1)

    for i, sample in enumerate(data):
        print(sample['mask'].max())


if __name__ == '__main__':
    main()
