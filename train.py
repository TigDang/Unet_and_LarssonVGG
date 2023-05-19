import argparse
from datetime import datetime

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from models.vgg16 import LarssonVGG16
from models.unet import UNet
from graycoco_dataset import train_dataset, val_dataset
from utils.utils import save_fig, plot_KLD, get_KLD, get_val_loss, load_obj, save_obj, save_artifacts, kaiming_init, print_model_summary

parser = argparse.ArgumentParser(
                    prog='Train colorization nn',
                    description='This script runs training of colorization neural network',
                    )

parser.add_argument('--name', help='defines name of experiment', default='testrun')
parser.add_argument('--epochs', type=int, help='how much training epochs need', default=100)
parser.add_argument('-si', '--save_iters', type=int, help='how much iters will past before saving metrics and etc', default=100)
parser.add_argument('-lr', '--learning_rate', type=float, help='defines learning rate', default=1e-5)
parser.add_argument('-bs', '--batch_size', type=int, help='how much training examples will be used per training iteration', default=20)
parser.add_argument('-m', '--modify', help='defines if modification will be used',
                    action='store_true')
parser.add_argument('--continue_train', help='defines if train will be started from last save',
                    action='store_true')
parser.add_argument('--device', default='cuda:0', help='defines which c.u. will used; note, that multy-gpu not allowed')
parser.add_argument('--target_net', default='None', help='defines name of model which will be used for comparing with modification')
parser.add_argument('--target_epoch', default='latest', help='defines which version of target net use')


if __name__=='__main__':
    opt = parser.parse_args()

    now = datetime.now()
    dt_string = now.strftime("(%d_%m_%Y %H_%M_%S)")
    opt.name += dt_string

    train_losses = []
    val_losses = []

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
    softmax = nn.Softmax(dim=1)

    # Define if model will be pretrained
    if opt.continue_train:
        model = load_obj(opt, 'nets', 'latest')
        train_losses, val_losses = load_obj(opt, 'losses', 'latest')
    else:
        # Define which model will be used - Unet or VGG
        if opt.modify:
            model = UNet(1, 1024)
            vgg_model = load_obj(opt, 'nets', opt.target_epoch, opt.target_net)
            kld_losses = []
        else:
            model = LarssonVGG16()
            
        model.apply(kaiming_init)
    
    print_model_summary(model, (1, 64, 64))

    model = model.float().to(opt.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # Define dataloader, which uses to load batches of images
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)


    # Epochs loop
    for epoch in range(opt.epochs):
        print(f'\n  epoch {epoch} had started')

        # loop over the dataset multiple times
        for i, data in enumerate(train_loader, 0):
            
            # Check and save metrics
            if i%opt.save_iters == 1:
                save_artifacts(opt, model, f'e_{epoch} i_{i}', val_loader, (train_losses, val_losses))
                val_losses.append(get_val_loss(opt, val_loader, model, loss_function))
                train_losses.append(loss.item())

                # For modification metrics
                if opt.modify:
                    kld_losses.append(get_KLD(opt, vgg_model, model, val_loader))
                    save_obj(opt, kld_losses, 'KLD_losses', 'latest')
                    save_fig(plot_KLD(kld_losses), opt, 'KLD_losses', 'latest')
                    
                

            # load batch
            l, ab = data
            l, ab = l.float().to(opt.device), ab.long().to(opt.device)
        
            # making the model parameters gradients
            optimizer.zero_grad()
        
            # generate colorization and train
            # forward + backward + optimize
            outputs_ab = model(l)
            
            # Computing a* and b* losses separetely
            loss = loss_function(softmax(outputs_ab[:, :512]), ab[:, 0]) + loss_function(softmax(outputs_ab[:, 512:]), ab[:, 1])
            
            # Computing gradients for every parameter used in loss computing
            loss.backward()

            # Making gradiend step for models parameters 
            optimizer.step()
        
            # train_losses.append(loss.item())
            print(f'    iter {i} loss: {loss}')
        
        print(f'    end of epoch, ')

        ## save model
        save_obj(opt, model, 'nets', f'{epoch}')
