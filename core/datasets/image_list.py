from torchvision.datasets import VisionDataset
import warnings
import torch
from PIL import Image
import os
import os.path
import numpy as np
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ImageList(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root=None, transform=None, target_transform=None,  empty=False):
        super(ImageList, self).__init__(root, transform=transform, target_transform=target_transform)

        self.empty = empty
        if empty:
            self.samples = np.empty((1, 2), dtype='<U1000')
        else:
            self.samples = np.loadtxt(root, dtype=np.dtype((np.unicode_, 1000)), delimiter=' ')
        self.loader = pil_loader

        self.identity = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        path, label = self.samples[index]
        label = int(label)

        output = {
            'label': label,
            'path': path,
            'index': index
        }

        img0 = self.loader(path)

        # original image without transform
        output['img'] = self.identity(img0)

        if self.transform is not None:
            output['img0'] = self.transform(img0)

        return output

    def __len__(self):
        return len(self.samples)

    def add_item(self, addition):
        if self.empty:
            self.samples = addition
            self.empty = False
        else:
            self.samples = np.concatenate((self.samples, addition), axis=0)
        return self.samples

    def remove_item(self, reduced):
        self.samples = np.delete(self.samples, reduced, axis=0)
        return self.samples
