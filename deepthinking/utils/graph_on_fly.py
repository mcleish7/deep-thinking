""" fcifar10_data.py
    CIFAR-10 related dataloaders

    based on mazes_data.py
"""

import torch
from torch_geometric import data
from .graph_on_fly_class import Identity, ConnectedComponents, ShortestPath
import numpy as np

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_identity_graph_loader(train_batch_size, test_batch_size, split, rank, vary=False, shuffle=True):

    traindata = Identity(split, rank, vary)
    testset = Identity(split, rank, vary)

    train_split = int(0.8 * len(traindata))

    trainset, valset = torch.utils.data.random_split(traindata,
                                                     [train_split,
                                                      int(len(traindata) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
