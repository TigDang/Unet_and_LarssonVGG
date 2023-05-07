from glob import glob
import os
from os import listdir
from os.path import isfile, join

import torch
import torchvision
from torchvision import transforms
import PIL
import cv2
from matplotlib import pyplot as plt

from transformations.train_transformations import CropToSquare
from utils.utils import convert_rgb2lab


class GrayCocoDataset():
    def __init__(self, root, mode, transform=None):
        self.root = f'{root}/{mode}/'
        self.transform = transform
        self.mode = mode

        try:
            self.paths = glob(os.path.join(self.root, "*.jpg"))
        except FileNotFoundError:
            print(f'There is no data in such path: {self.root}')

    def __len__(self):
        return self.paths.__len__()
    
    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        image = convert_rgb2lab(image)
        image = torchvision.transforms.ToTensor()(image)

        if self.transform:
            image = self.transform(image)
       

        l = image[0:1]/100      # [0, 100] -> [0, 1]
        ab = image[1:]/256+0.5  # [-128, 128] -> [0, 1] 

        return l, ab
    

datapath = './data/coco'
transform = transforms.Compose([
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.RandomCrop((375, 375)),
    CropToSquare(),
    transforms.Resize((512, 512)),
])

train_dataset = GrayCocoDataset(datapath, 'train', transform=transform)
val_dataset = GrayCocoDataset(datapath, 'val', transform=transform)
test_dataset = GrayCocoDataset(datapath, 'test', transform=transform)


if __name__== '__main__':
    # Some tests
    print(val_dataset.__len__())

    from utils.utils import convert_lab2rgb
    for i in range(5):
        l, ab = val_dataset.__getitem__(i)
        lab_image = torch.concat((l, ab))
        rgb_image = convert_lab2rgb(lab_image.permute(1, 2, 0).numpy())
        # rgb_image = lab2rgb(lab_image.view(l.shape[1], l.shape[2], 3))
        # rgb_image = rgb_image.resize(2, 0, 1)
        # cv2.imshow('some', rgb_image)
        plt.imshow(rgb_image)
        plt.show()
