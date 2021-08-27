import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import torchvision.transforms.functional as F
import SimpleITK as sitk
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import json
plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


"""
dog1_int = read_image(str(Path('assets') / 'dog1.jpg'))
dog2_int = read_image(str(Path('assets') / 'dog2.jpg'))

grid = make_grid([dog1_int, dog2_int, dog1_int, dog2_int])
show(grid)



boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430]], dtype=torch.float)
colors = ["blue", "yellow"]
result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
show(result)"""
"""image = sitk.ReadImage("D:/chest_xray/train/NORMAL/NORMAL2-IM-0986-0001.jpeg")
print(sitk.GetArrayFromImage(image))
plt.imshow(sitk.GetArrayFromImage(image),cmap='Greys')
plt.show()"""

# read the image
"""im: Image = Image.open("D:/Car-dataset/train/0cdf5b5d0ce1_01.jpg")

np_im = np.array(im)

im1: Image = Image.open("D:/Car-dataset/Mask_train/0cdf5b5d0ce1_01_mask.gif")

np_im1 = np.array(im1)
print(np.shape(np_im1), np.shape(np_im))
plt.imshow(np_im, alpha=0.1)
plt.imshow(np_im1, cmap="Greys", alpha=0.5)
plt.show()"""
# show image
"""dog1_int = read_image("C:/Users/alexn/Desktop/PetImages/test1/21.jpg")
grid = make_grid([dog1_int, dog1_int, dog1_int, dog1_int])

batch_int = torch.stack([dog1_int, dog1_int])
batch = convert_image_dtype(batch_int, dtype=torch.float)
print(dog1_int.size())
print(type(dog1_int))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
t = torchvision.models.video.r3d_18()
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(batch)
# print(predictions)
score_threshold = .8
dogs_with_boxes = [
    draw_bounding_boxes(dog_int, boxes=output['boxes'][output['scores'] > score_threshold], width=4)
    for dog_int, output in zip(batch_int, predictions)
]
for output in predictions:
    print(output['labels'])
show(dogs_with_boxes)"""
data = open("/SemanticSegmentation/Experiment/config_seg_.json")
data =json.load(data)
print(data['learning_rate'])