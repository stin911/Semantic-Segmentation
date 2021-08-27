import os, random
import numpy as np
import shutil
import torch


def split_data(main_folder, mask_folder, val_image_folder=None, val_mask_folder=None):
    number_sample = int((len(os.listdir(main_folder)) * 10) / 100)
    list_moved = []
    for i in range(number_sample - 1):
        file = random.choice(os.listdir(main_folder))
        if file not in list_moved:
            list_moved.append(file)

            shutil.move(main_folder + file, val_image_folder + file)
            file_mask = file[:-4] + "_mask.gif"

            shutil.move(mask_folder + file_mask, val_mask_folder + file_mask)




