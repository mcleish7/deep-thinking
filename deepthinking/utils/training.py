""" training.py
    Utilities for training models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from dataclasses import dataclass
from random import randrange

import torch
from icecream import ic
from tqdm import tqdm
import sys

from deepthinking.utils.testing import get_predicted


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"


def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k


def train(net, loaders, mode, train_setup, device):
    if mode == "progressive":
        train_loss, acc = train_progressive(net, loaders, train_setup, device)
    if mode == "gnn":
        train_loss, acc = train_gnn(net, loaders, train_setup, device)
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return train_loss, acc


def train_progressive(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        # print("after")
        # print(targets)
        # print(targets.shape)
        targets = targets.view(targets.size(0), -1)
        # print("after")
        # print(targets)
        # print(targets.shape)
        if problem == "mazes":
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        # print(inputs)
        # print(inputs.shape)
        outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes":
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        # print("loss max iters is: ",loss_max_iters_mean)
        # print("loss progressive is: ",loss_progressive_mean)
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        # print("loss item is:",loss.item())
        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

        # from sklearn.metrics import mean_squared_error
        # mse = round(mean_squared_error(predicted,targets),2)
        
    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc


def train_gnn(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs) in enumerate(tqdm(trainloader, leave=False)):
        inputs = inputs.to(device)
        targets = inputs.y.long().to(device)

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).

        # print("max iters is ",max_iters)
        # print("inputs is type ",type(inputs))
        # print("inputs x shape is ",inputs.x.shape)
        # print("inpouts edges shape is ", inputs.edge_index.shape)
        # print("Inputs edge attributes is ", inputs.edge_attr.shape)
        # print("inputs y is ", inputs.y.shape)


        preds = []
        state = None
        for i in range(max_iters):
            pred, state = net(inputs, state)
            preds.append(pred)
        outputs_max_iters = torch.stack(preds, dim=1)
        # outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)


        # print(type(targets.edge_index))
        # if alpha != 1:
        #     print("max iter shape before is ",outputs_max_iters.shape)
        #     outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
        #                                                outputs_max_iters.size(1), -1)
            #  print("output_max_iters is a ",type(outputs_max_iters))
        # print("output_max_iters shape is ",outputs_max_iters.shape)
        # print("targets shape is ",targets.shape)
        # sys.exit()
        #     loss_max_iters = criterion(outputs_max_iters, targets)

        # else:
        #     loss_max_iters = torch.zeros_like(targets).float()
        # # print("loss is ",loss_max_iters)
        # # sys.exit()

        # # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # # so we save time by setting it equal to 0).
        # if alpha != 0:
        #     outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
        #     outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        #     loss_progressive = criterion(outputs, targets)
        # else:
        #     loss_progressive = torch.zeros_like(targets).float()

        # loss_max_iters_mean = loss_max_iters.mean()
        # loss_progressive_mean = loss_progressive.mean()

        # loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        # print("loss max iters is: ",loss_max_iters_mean)
        # print("loss progressive is: ",loss_progressive_mean)

        # im_loss = torch.pow(preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
        im_loss = torch.pow(outputs_max_iters[:, -1:] - targets[:, None, :], 2).mean(dim=-1).mean(dim=-1)
        # im_loss = torch.linalg.norm(targets - outputs_max_iters)
        loss = im_loss.mean()
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        # print("loss item is:",loss.item())
        train_loss += loss.item()
        # predicted = get_predicted(inputs, outputs_max_iters, problem)
        predicted = outputs_max_iters[:, -1:]
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)
        # print("corrects is ", correct)
        # print("targets shape is", targets.shape)
        # print('targets size 0 is ',targets.size(0))
        # sys.exit()

        # from sklearn.metrics import mean_squared_error
        # mse = round(mean_squared_error(predicted,targets),2)
        
    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc