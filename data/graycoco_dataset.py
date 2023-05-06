import torchvision
import PIL
from skimage.color import rgb2lab, lab2rgb
from os import listdir
from os.path import isfile, join



class GrayCocoDataset():
    def __init__(self, root, mode):
        root = f'{root}/{mode}/)'

        try:
            self.paths = [f for f in listdir(root) if isfile(join(root, f))]
        except FileNotFoundError:
            print('There is no data in such path')

    
    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = PIL.Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = rgb2lab(image)
        image = torchvision.transforms.ToTensor()(image)

        return image[0], image[1:]

train_dataset = GrayCocoDataset('.', 'train')
val_dataset = GrayCocoDataset('.', 'val')
test_dataset = GrayCocoDataset('.', 'test')


# TODO: make test
if __name__== '__main__':
    pass
