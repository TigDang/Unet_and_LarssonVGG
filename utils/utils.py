import os
import torch
from torch import nn
from torchsummary import summary
from skimage import color
from skimage.color import rgb2lab as convert_rgb2lab
import numpy as np
from matplotlib import pyplot as plt
import numpy as np

def print_model_summary(model, input_size):
    summary(model, input_size)

def save_obj(opt, object, folder, label):
    save_path = f'./artifacts/{folder}/{opt.name}_{label}.pt'
    print(f'    saving object at {save_path}')
    torch.save(object, save_path)


def load_obj(opt, folder, label):
    load_path = f'./artifacts/{folder}/{opt.name}_{label}.pt'
    model = torch.load(load_path, map_location=opt.device)
    return model

def convert_lab2rgb(lab):
   lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
#    lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
   lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] - 128, -128, 128)  # [0, 256] -> [-128, 128]
   rgb = color.lab2rgb( lab.astype(np.float64) )
   return rgb

def save_val_images(opt, loader, iter, model):
        from utils.utils import convert_lab2rgb
        i, data = next(enumerate(loader, 0))
        l, ab = data
        l, ab = l.float().to(opt.device), ab.long()
        outputs_ab = model(l).detach().cpu()
        outputs_a, outputs_b = torch.argmax(outputs_ab[0:1, :512], dim=1), torch.argmax(outputs_ab[0:1, 512:], dim=1)
        outputs_ab = torch.concat((outputs_a, outputs_b), dim=0)

        l = l.detach().cpu()

        lab_true_image = torch.concat((l[0], ab[0]))
        rgb_true_image = convert_lab2rgb(lab_true_image.permute(1, 2, 0).numpy())

        lab_generated_image = torch.concat((l[0], outputs_ab))
        rgb_generated_image = convert_lab2rgb(lab_generated_image.permute(1, 2, 0).numpy())

        fig = plt.imshow(np.concatenate(((torch.concat((l, l, l), dim=1))[0].permute(1, 2, 0).numpy(), rgb_true_image, rgb_generated_image), axis=1))
        
        fig.set_label(iter)

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        try:
            plt.savefig(f'./artifacts/images/{opt.name}/{iter}.jpg', bbox_inches='tight')
        except FileNotFoundError:
            os.mkdir(f'./artifacts/images/{opt.name}/')
            plt.savefig(f'./artifacts/images/{opt.name}/{iter}.jpg', bbox_inches='tight')

def kaiming_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

def get_grad_distr(model, layer_types = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    weights = [module.weight.grad.detach().cpu().data.flatten() for module in model.modules() if isinstance(module, layer_types)]
    labels = [module.__str__() for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d)]
    plt.figure(figsize=(8, 10), dpi=80)
    plt.grid(visible=True, axis='y')
    plt.boxplot(labels=labels, x=weights)
    plt.title('Distribution of convolutions gradients')
    plt.ylabel('Amount of grad')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    return plt.figure

def plot_losses(losses):
    plt.plot(losses, 'red')
    plt.title('Значение функции ошибки во время тренировки')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    return plt.figure

def save_fig(fig, opt, foldername, label):
    try:
        plt.savefig(f'./artifacts/{foldername}/{opt.name}/{label}.jpg', bbox_inches='tight')
    except FileNotFoundError:
        os.mkdir(f'./artifacts/{foldername}/{opt.name}/')
        plt.savefig(f'./artifacts/{foldername}/{opt.name}/{label}.jpg', bbox_inches='tight')
    plt.close()


def save_artifacts(opt, model, label, loader, losses):
    save_fig(get_grad_distr(model), opt, 'grad_distrs', label)
    save_fig(plot_losses(losses), opt, 'loss_plots', label)
    save_obj(opt, model, 'nets', label)
    save_obj(opt, losses, 'losses', label)
    save_val_images(opt, loader, label, model)

    