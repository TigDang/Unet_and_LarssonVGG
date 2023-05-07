import torch
from torchvision import transforms

class CropToSquare(torch.nn.Module):
    """
    # Implements cropping to minimal shape
    # to make square image from an any image
    """
    
    def forward(self, img):
        min_shape = min(img.shape[1], img.shape[2])
        return transforms.functional.crop(img, 0, 0, min_shape, min_shape)
    