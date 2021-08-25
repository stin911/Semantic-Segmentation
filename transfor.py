import torchvision.transforms.functional as F
import numpy as np
import torch
import PIL.Image as Image


class ResizeMine:
    def __call__(self, sample):
        image = F.resize(sample['image'], [512, 512], interpolation=Image.BILINEAR)
        mask = F.resize(sample['mask'], [512, 512], interpolation=Image.NEAREST)

        sample = {'image': image, 'mask': mask}
        return sample
class ToTensor:
    def __call__(self,sample):
        image = np.array(sample['image'])
        mask = np.array(sample['mask'], np.float32)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        image = image.permute(2, 0, 1)
        sample = {'image': image, 'mask': mask}
        return sample


