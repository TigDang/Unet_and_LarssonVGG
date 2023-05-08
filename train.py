import argparse

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from models.vgg16 import LarssonVGG16
from graycoco_dataset import train_dataset, val_dataset
from utils.utils import load_obj, save_obj, save_artifacts, kaiming_init, print_model_summary

parser = argparse.ArgumentParser(
                    prog='Train colorization nn',
                    description='This script runs training of colorization neural network',
                    )

parser.add_argument('--name', help='defines name of experiment', default='testrun')
parser.add_argument('--epochs', type=int, help='how much training epochs need', default=10)
parser.add_argument('-lr', '--learning_rate', type=float, help='defines learning rate', default=1e-4)
parser.add_argument('-bs', '--batch_size', type=int, help='how much training examples will be used per training iteration', default=1)
parser.add_argument('-m', '--modify', help='defines if modification will be used',
                    action='store_true')
parser.add_argument('--continue_train', help='defines if train will be started from last save',
                    action='store_true')
parser.add_argument('--device', default='cpu', help='defines which c.u. will used; note, that multy-gpu not allowed')

if __name__=='__main__':
    opt = parser.parse_args()

    losses = []

    loss_function = nn.CrossEntropyLoss()

    # Define if model will be pretrained
    if opt.continue_train:
        model = load_obj(opt)
    else:
        # Define which model will be used - Unet or VGG
        if opt.modify:
            pass # TODO: Need a Unet
        else:
            model = LarssonVGG16().float().to(opt.device)
        model.apply(kaiming_init)
    
    print_model_summary(model, (1, 256, 256))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # Define dataloader, which uses to load batches of images
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    for epoch in range(opt.epochs):
        print(f'\n  epoch {epoch} had started')

        # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            if i%10 == 1:
                save_artifacts(opt, model, f'e_{epoch} i_{i}', val_loader, losses)
                

            # load batch
            l, ab = data
            l, ab = l.float().to(opt.device), ab.long().to(opt.device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # generate colorization and trainc
            # forward + backward + optimize
            outputs_ab = model(l)
            loss = loss_function(outputs_ab[:, :512], ab[:, 0]) + loss_function(outputs_ab[:, 512:], ab[:, 1])
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            losses.append(loss.item())
            print(f'    iter {i} loss: {loss}')
        
        print(f'    end of epoch, ')
        print('     Loss: {}'.format(running_loss))

    

        ## save model
        save_obj(opt, model, f'{epoch}')