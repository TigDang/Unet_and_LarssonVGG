import argparse
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt

from graycoco_dataset import test_dataset
from utils.utils import save_fig, load_obj, convert_lab2rgb, convert_rgb2lab

parser = argparse.ArgumentParser(
                    prog='Test colorization nn',
                    description='This script runs testing of chosed colorization neural network',
                    )

parser.add_argument('--name', default='Unet_run_tl_en4(15_05_2023 12_24_51)',  help='defines name of experiment which will be tested')
parser.add_argument('--epoch', help='defines epoch which will be tested', default='latest')
parser.add_argument('-bs', '--batch_size', type=int, help='how much training examples will be used per training iteration', default=1)
parser.add_argument('--device', default='cpu', help='defines which c.u. will used; note, that multy-gpu not allowed')
parser.add_argument('-m', action='store_true', help='defines, is modification will be used', default=True)
# if -m, then:
parser.add_argument('--target_name', default='VGG_Varya_again(13_05_2023 22_30_43)',  help='defines name of experiment which will be tested')
parser.add_argument('--target_epoch', help='defines epoch which will be tested', default='2')


if __name__=='__main__':
    opt = parser.parse_args()

    model = load_obj(opt, 'nets', opt.epoch, opt.name)

    if opt.m:
        target_model = load_obj(opt, 'nets', opt.target_epoch, opt.target_name)

    testloader = DataLoader(test_dataset, opt.batch_size, shuffle=True)

    with torch.no_grad():
        for iter, data in enumerate(testloader):
            l, ab = data
            l, ab = l.float().to(opt.device), ab.long()
            outputs_ab = model(l).detach().cpu()
            outputs_a, outputs_b = torch.argmax(outputs_ab[0:1, :512], dim=1), torch.argmax(outputs_ab[0:1, 512:], dim=1)
            
            # normalization
            outputs_a = (outputs_a - outputs_a.float().mean())*5 + 128
            outputs_b = (outputs_b - outputs_b.float().mean())*10 + 128
            outputs_ab = torch.concat((outputs_a, outputs_b), dim=0)

            if opt.m:
                target_outputs_ab = target_model(l).detach().cpu()
                target_outputs_a, target_outputs_b = torch.argmax(target_outputs_ab[0:1, :512], dim=1), torch.argmax(target_outputs_ab[0:1, 512:], dim=1)
                target_outputs_ab = torch.concat((target_outputs_a, target_outputs_b), dim=0)
                target_outputs_ab = (target_outputs_ab - target_outputs_ab.float().mean()) + 128
                target_lab_generated_image = torch.concat((l[0], target_outputs_ab))
                target_rgb_generated_image = convert_lab2rgb(target_lab_generated_image.permute(1, 2, 0).numpy())

            l = l.detach().cpu()

            lab_true_image = torch.concat((l[0], ab[0]))
            rgb_true_image = convert_lab2rgb(lab_true_image.permute(1, 2, 0).numpy())

            lab_generated_image = torch.concat((l[0], outputs_ab))
            rgb_generated_image = convert_lab2rgb(lab_generated_image.permute(1, 2, 0).numpy())

            if opt.m:
                fig = plt.imshow(np.concatenate(((torch.concat((l, l, l), dim=1))[0].permute(1, 2, 0).numpy(), rgb_true_image, target_rgb_generated_image, rgb_generated_image), axis=1))
            else:
                fig = plt.imshow(np.concatenate(((torch.concat((l, l, l), dim=1))[0].permute(1, 2, 0).numpy(), rgb_true_image, rgb_generated_image), axis=1))
            
            fig.set_label(iter)

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)


            try:
                plt.savefig(f'./artifacts/results/{opt.name}/{iter}.png', bbox_inches='tight')
            except FileNotFoundError:
                os.mkdir(f'./artifacts/results/{opt.name}/')
                plt.savefig(f'./artifacts/results/{opt.name}/{iter}.png', bbox_inches='tight')
            plt.close()

