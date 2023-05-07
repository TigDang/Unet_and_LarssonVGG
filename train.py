import argparse

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from models.vgg16 import LarssonVGG16
from graycoco_dataset import train_dataset, val_dataset
from utils.utils import load_model, save_model

parser = argparse.ArgumentParser(
                    prog='Train colorization nn',
                    description='This script runs training of colorization neural network',
                    )

parser.add_argument('--name', help='defines name of experiment')
parser.add_argument('--epochs', type=int, help='how much training epochs need', default=10)
parser.add_argument('-bs', '--batch_size', type=int, help='how much training examples will be used per training iteration', default=1)
parser.add_argument('-m', '--modify', help='defines if modification will be used',
                    action='store_true')
parser.add_argument('--contunue', help='defines if train will be started from last save',
                    action='store_true')
parser.add_argument('--device', default='cuda:0', help='defines which c.u. will used; note, that multy-gpu not allowed')

if __name__=='__main__':
    opt = parser.parse_args()

    loss_function = nn.CrossEntropyLoss()

    # Define which model will be used - Unet or VGG
    if opt.modify:
        pass
    else:
        model = LarssonVGG16()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Define dataloader, which uses to load batches of images
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)

    for epoch in range(opt.epochs):
        ## load batch

        ## generate colorization and trainc

        ## check metrics

        ## save model
        pass