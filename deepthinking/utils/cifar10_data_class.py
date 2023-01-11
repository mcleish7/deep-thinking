""" based on classes from easy_to_hard_data.py
"""
import errno
import os
import os.path
import tarfile
import urllib.request as ur
from typing import Optional, Callable

import numpy as np
import torch
from tqdm import tqdm
import shutil
import torchvision
import torchvision.transforms as transforms

class CIFARDataset(torch.utils.data.Dataset):
    """This is a dataset class for CIFAR-10.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 download: bool = True):

        self.root = root
        self.train = train
        if transform is None:
            # print("triggered")
            self.transform = transform = transforms.Compose(
                                    [transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #taken from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        else:
            self.transform = transform
        # print(type(self.root))
        # print(type(self.train))
        # print(type(self.transform))
        print(f"Loading CIFAR-10 data")
        if download:
            # trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
            self.data = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=True, transform=self.transform)
        else:
            self.data = torchvision.datasets.CIFAR10(root=self.root, train=self.train, transform=self.transform, download=False)

    def __getitem__(self, index):
        img, target = self.data[index]
        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root,"cifar-10-batches-py")
        if not os.path.exists(fpath):
            return False
        return True


if __name__ == "__main__":
    gd = CIFARDataset("./data")
    print("CIFAR-10 data loaded.")
