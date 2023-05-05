import torchvision
import PIL
from skimage.color import rgb2lab, lab2rgb


class GraySun(torchvision.datasets.SUN397):
    # Override parent __getitem__ method - now it return l* and a*b* channels
    
    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        image = PIL.Image.open(image_file).convert("RGB")


        if self.transform:
            image = self.transform(image)


        image = rgb2lab(image)
        image = torchvision.transforms.ToTensor()(image)


        return image[0], image[1:]

graysun_dataset = GraySun('.', download=True)


# TODO: make test
if __name__== '__main__':
    pass
