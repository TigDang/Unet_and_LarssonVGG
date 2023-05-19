import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchsummary import summary
from skimage import color
from skimage.color import rgb2lab as convert_rgb2lab


def get_val_loss(opt, val_loader, model, loss_function):
    """Calculate validation loss

    Args:
        opt (namespace): defines device which will be used for computing
        val_loader (torch.utils.data.DataLoader): used for take first batch for calculation
        model (nn.Module): neural net which will be used for calculate output
        loss_function (torch.nn.CrossEntropyLoss): defines loss function which will be used for calculate

    Returns:
        (numpy): validation loss
    """    

    # Gets input and output
    i, data = next(enumerate(val_loader, 0))
    l, ab = data
    l, ab = l.float().to(opt.device), ab.long().to(opt.device)
    softmax = nn.Softmax(dim=1)
    outputs_ab = model(l)

    # Calculate a* and b* losses separately and sum them
    loss = loss_function(softmax(outputs_ab[:, :512]), ab[:, 0]) + loss_function(softmax(outputs_ab[:, 512:]), ab[:, 1])
    
    return loss.data.detach().cpu().numpy()

def print_model_summary(model, input_size):
    """Prints summary of model layers

    Args:
        model (_type_): a model which summary will be printed
        input_size (tuple): C-H-W tuple of test input size which will be used to calculate summary
    """    

    summary(model, input_size, device='cpu')

def save_obj(opt, object, folder, label):
    try:
        os.mkdir(f'./artifacts/{folder}/{opt.name}')
    except FileExistsError:
        pass

    
    save_path = f'./artifacts/{folder}/{opt.name}/{label}.pt'
    
    print(f'    saving object at {save_path}')
    torch.save(object, save_path)

def load_obj(opt, folder, label = None, name=None):
    if name==None:
        name = opt.name

    if label==None:
        label = 'latest'

    load_path = f'./artifacts/{folder}/{name}/{label}.pt'
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
            plt.savefig(f'./artifacts/images/{opt.name}/{iter}.png', bbox_inches='tight')
        except FileNotFoundError:
            os.mkdir(f'./artifacts/images/{opt.name}/')
            plt.savefig(f'./artifacts/images/{opt.name}/{iter}.png', bbox_inches='tight')
        plt.close()

def kaiming_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, a=0.01)
        

def get_grad_distr(model, layer_types = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
    weights = [module.weight.grad.detach().cpu().data.flatten()[:1000] for module in model.modules() if isinstance(module, layer_types)]
    labels = [module.__str__() for module in model.modules() if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d)]
    plt.figure(figsize=(8, 10), dpi=80)
    plt.grid(visible=True, axis='y')
    plt.boxplot(labels=labels, x=weights)
    plt.title('Distribution of convolutions gradients')
    plt.ylabel('Amount of grad')
    plt.ylim(-0.001, 0.001)
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.show()
    return plt.figure

def plot_losses(losses):
    train_loss, val_loss = losses
    plt.plot(train_loss, 'red')
    plt.plot(val_loss, 'blue')
    plt.title('Значение функции ошибки во время тренировки')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка')
    plt.legend(['train', 'val'])
    return plt.figure

def plot_KLD(loss):
    plt.plot(loss, 'green')
    plt.title('Значение дивергенции Кульбака-Лейблера')
    plt.xlabel('Итерация')
    plt.ylabel('Дивергенция в ср. битах')
    return plt.figure

def save_fig(fig, opt, foldername, label):
    path = f'./artifacts/{foldername}/{opt.name}/{label}.svg'
    try:
        plt.savefig(path, bbox_inches='tight')
    except FileNotFoundError:
        os.mkdir(f'./artifacts/{foldername}/{opt.name}')
        plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_artifacts(opt, model, label, loader, losses):
    save_fig(get_grad_distr(model), opt, 'grad_distrs', label)
    save_fig(get_grad_distr(model), opt, 'grad_distrs', 'latest')
    save_fig(plot_losses(losses), opt, 'loss_plots', label)
    save_fig(plot_losses(losses), opt, 'loss_plots', 'latest')
    save_obj(opt, model, 'nets', 'latest')
    save_obj(opt, losses, 'losses', 'latest')
    save_val_images(opt, loader, label, model)
    save_val_images(opt, loader, 'latest', model)

def get_KLD(opt, target_net, totest_net, val_loader):
    """Calculate mean (per hist) Kullback Leibler divergence between generated hists from totest_net to target_net.

    Args:
        opt (namespace): defines device which will be used for computing
        target_net (nn.Module): neural network to which generated hists you want to calculate KLD
        totest_net (nn.Module): neural network which from generated hists you want to calculate KLD
        val_loader (torch.utils.data.DataLoader): used for take first batch for calculation

    Returns:
        KLD (numpy): mean of KLD from totest_net hists to target_net hists
    """    

    softmax = nn.Softmax(dim=1)

    # Gets input and output
    i, data = next(enumerate(val_loader, 0))
    l, ab = data
    l, ab = l.float().to(opt.device), ab.long().to(opt.device)

    target_out = target_net(l)
    totest_out = totest_net(l)

    # Making probability hists from outputs of net
    target_hist_a = softmax(target_out[:, :512])
    target_hist_b = softmax(target_out[:, 512:])

    totest_hists_a = softmax(totest_out[:, :512])
    totest_hists_b = softmax(totest_out[:, 512:])

    # Concatenate a* and b* hists
    target_hists = torch.concat((target_hist_a, target_hist_b), dim=2)
    target_hists = target_hists.flatten(start_dim=2, end_dim=- 1).flatten(start_dim=0, end_dim=1)   

    totest_hists = torch.concat((totest_hists_a, totest_hists_b), dim=2)
    totest_hists = totest_hists.flatten(start_dim=2, end_dim=- 1).flatten(start_dim=0, end_dim=1)

    # Calculate KLD
    div_KL = torch.sum(target_hists*torch.log2(target_hists/totest_hists), dim=0)
    return (torch.mean(div_KL)/2).detach().cpu().numpy()

if __name__ == '__main__':
    pass
