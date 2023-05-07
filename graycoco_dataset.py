from glob import glob
import os
from os import listdir
from os.path import isfile, join

import torch
import torchvision
import PIL
import cv2
from matplotlib import pyplot as plt
from skimage.color import rgb2lab


class GrayCocoDataset():
    def __init__(self, root, mode, transform=None):
        self.root = f'{root}/{mode}/'
        self.transform = transform

        try:
            self.paths = glob(os.path.join(self.root, "*.jpg"))
        except FileNotFoundError:
            print(f'There is no data in such path: {self.root}')

    def __len__(self):
        return self.paths.__len__()
    
    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = rgb2lab(image)
        image = torchvision.transforms.ToTensor()(image)

        return image[0:1], image[1:]
    

datapath = './data/coco'

train_dataset = GrayCocoDataset(datapath, 'train')
val_dataset = GrayCocoDataset(datapath, 'val')
test_dataset = GrayCocoDataset(datapath, 'test')


if __name__== '__main__':
    from utils.utils import convert_lab2rgb
    for i in range(1):
        l, ab = val_dataset.__getitem__(i)
        lab_image = torch.concat((l, ab))
        rgb_image = convert_lab2rgb(lab_image.numpy())
        # rgb_image = lab2rgb(lab_image.view(l.shape[1], l.shape[2], 3))
        # rgb_image = rgb_image.resize(2, 0, 1)
        # cv2.imshow('some', rgb_image)
        plt.imshow(rgb_image)
        plt.show()
