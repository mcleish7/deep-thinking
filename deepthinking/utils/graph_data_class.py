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


def get_file(from_folder, to_folder): #input is the folder whihc contains the data and solutions and the folder we want to move it to
    dir = os.path.join("~/Desktop/graph_generation_files", from_folder)
    dir = os.path.expanduser(dir)
    path = os.path.join(to_folder, from_folder)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('Using existing file', from_folder)
        return path

    try:
        shutil.copytree(dir, path)
        print(f"data copied from {dir} to {path}")
    except:
        if os.path.exists(path):
             os.remove(path)
        raise RuntimeError('Stopped downloading due to interruption.')

    return path

class GraphDataset(torch.utils.data.Dataset):
    """This is a dataset class for Graphs.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 size: int = 6,
                 transform: Optional[Callable] = None,
                 download: bool = True):

        self.root = root
        self.train = train
        self.size = size
        self.transform = transform

        self.folder_name = f"graph_{'train' if self.train else 'test'}_{size}"

        if download:
            self.download(self.folder_name)

        print(f"Loading graphs with {size} nodes")

        inputs_path = os.path.join(root, self.folder_name, "inputs.npz")
        solutions_path = os.path.join(root, self.folder_name, "solutions.npz")

        inputs_np = np.load(inputs_path)['arr_0']
        targets_np = np.load(solutions_path)['arr_0']
        
        self.inputs = torch.from_numpy(inputs_np).unsqueeze(1).float()
        self.targets = torch.from_numpy(targets_np).long()

    def __getitem__(self, index):
        img, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            stacked = torch.cat([img, target.unsqueeze(0)], dim=0)
            stacked = self.transform(stacked)
            img = stacked[:3].float()
            target = stacked[3].long()

        return img, target

    def __len__(self):
        return self.inputs.size(0)

    def _check_integrity(self) -> bool:
        root = self.root
        fpath = os.path.join(root, self.folder_name)
        if not os.path.exists(fpath):
            return False
        return True

    def download(self, folder_name) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        get_file(folder_name, self.root)


if __name__ == "__main__":
    gd = GraphDataset("./data")
    print("Graph data loaded.")
