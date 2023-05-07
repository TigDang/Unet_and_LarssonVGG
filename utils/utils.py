import torch
from skimage import color
import numpy as np

def save_model(opt, model, epoch='latest'):
    save_path = f'artifacts\\nets\\{opt.name}_{epoch}.pt'
    torch.save(model, save_path)


def load_model(opt, epoch='latest'):
    load_path = f'artifacts\\nets\\{opt.name}_{epoch}.pt'
    model = torch.load(load_path, map_location=opt.device)
    return model

def convert_lab2rgb(lab):
   lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
   lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
   rgb = color.lab2rgb( lab.astype(np.float64) )
   return rgb