import torchvision.transforms.functional as F
import numpy as np
import torch
import PIL.Image as Image
from torchvision import transforms as T


class ResizeMine:
    def __call__(self, sample):
        image = F.resize(sample['image'], [512, 512], interpolation=T.InterpolationMode.BILINEAR)
        if sample['mask'] is not None:
            mask = F.resize(sample['mask'], [512, 512], interpolation=T.InterpolationMode.NEAREST)
        else:
            mask = sample['mask']
        sample = {'image': image, 'mask': mask}
        return sample


class CToTensor:
    def __call__(self, sample):
        image = np.array(sample['image'])

        mask = np.array(sample['mask'], np.float32)
        image = image.astype('float32')
        # normalize to the range 0-1
        image /= 255.0


        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        image = image.permute(2, 0, 1)


        sample = {'image': image, 'mask': mask}
        return sample
