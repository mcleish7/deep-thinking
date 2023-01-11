""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import einops
import torch
from icecream import ic
from tqdm import tqdm
import sys
import numpy as np

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(net, loaders, mode, iters, problem, device):
    accs = []
    for loader in loaders:
        if mode == "default":
            accuracy = test_default(net, loader, iters, problem, device)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)
        elif mode == "gnn":
            accuracy = test_gnn(net, loader, iters, problem, device)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs

def get_predicted(inputs, outputs, problem):
    if (problem == "graphs"):
        return outputs
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if (problem == "mazes"):# or (problem == "graphs"):
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)
    # elif problem == "graphs":
    #     # print("output is: ",outputs[0][0].flatten().unsqueeze(0).shape)
    #     # print("predicting: ",predicted.shape)
    #     print("inputs is ",inputs)
    #     print("outputs is: ",outputs)
    #     # print(outputs[0][0].flatten().unsqueeze(0))
    #     return outputs[0][0].flatten().unsqueeze(0)
    # print(predicted)
    return predicted


def test_default(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # #inputs = inputs.unsqueeze(0) #from stackoverflow but still not wokring
            # print(inputs.size())
            # inputs = inputs.unsqueeze(1)
            # inputs = inputs.view([128,4,4,125])
            # print(inputs.size())
            # #inputs = inputs.view(inputs.size(-1), -1) #copied from below

            all_outputs = net(inputs, iters_to_do=max_iters)
            # print("print all outputs size is",all_outputs.shape) #shape[batch_size,test_iters,2,dim,dim]
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]


                # torch.set_printoptions(profile="full")
                # print("outputs size is ",outputs.shape) #shape[batch_size,2,dim,dim]
                # #inputs = torch.reshape(inputs[0], (1,1, 512))
                # #print(outputs[0])
                # print("shape of big vector is ", outputs[0].shape) #shape[batch_size,dim,dim]
                predicted = get_predicted(inputs, outputs, problem)
                # print("predicted size is ",predicted.shape)
                # print(predicted[0])
                # print("shape of big vecotr is ", predicted[0].shape) #shape[dim,dim]
                # print("targets size is ",targets.shape)
                # torch.set_printoptions(profile="default")


                targets = targets.view(targets.size(0), -1)
                # print("targets size is ",targets.shape)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()
                # sys.exit()

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc

def test_gnn(net, testloader, iters, problem, device):
    # print("in gnn test")
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    dists = []
    print("testloader has length, ",len(testloader.dataset.indices))
    # print("dataloader 0 type is ", testloader[0])
    with torch.no_grad():
        for inputs in testloader:
            print("ran loop")
            inputs = inputs.to(device)
            targets = inputs.y.to(device)
            # print("type input is ", type(inputs))
            preds = []
            state = None
            for i in range(max_iters):
                pred, state = net(inputs, state)
                preds.append(pred)

            preds = torch.stack(preds, dim=0)
            # print("preds shape is ",preds.shape)
            # print("target shape is ", targets.shape)
            dist = (preds - targets[None, :])
            # print("dist shape is ",dist.shape)
            dist = torch.pow(dist, 2).mean(dim=-1) # square error
            # print("dist shape mean is ",dist.shape)
            # print("min is ",(torch.min(dist)))
            dist = dist.mean(dim=-1) # mean square error
            # print("dist is", dist)
            # print("dist shape mean mean is ",dist.shape)
            # print("min is ",(torch.min(dist)))
            dists.append(dist)
            # sys.exit()
            # n = dist.size(1)


            # # all_outputs = torch.stack(preds, dim=1)
            # for i in range(all_outputs.size(1)):
            #     outputs = all_outputs[:, i]
            #     # print("type outputs is ", type(outputs))
            #     # print("type targets is ", type(targets))
            #     # print("outputs shape is ",outputs.shape)
            #     # print("inputs x shape is ",inputs.x.shape)
            #     # print("inputs edge index shape is ",inputs.edge_index.shape)
            #     predicted = outputs.view(outputs.size(0), -1).to(device)
            #     # print("predicted shape is ",predicted.shape)
            #     targets = targets.view(targets.size(0), -1)
            #     # print("targets shape is ",targets.shape)

            #     # print("targets is ",targets)
            #     # print("predicted is ", predicted)
            #     # difference = targets-predicted
            #     # print(difference.shape)
            #     # print("difference is ",difference)
            #     # print("norm is ",torch.linalg.norm(targets-predicted).item())
            #     corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            # total += targets.size(0)

    # accuracy = 100.0 * corrects / total
    # print(dists)
    # print("accuracy shape is ",torch.stack(dists, dim=0).shape)
    # print(torch.stack(dists, dim=0))
    accuracy = torch.stack(dists, dim=0).mean(dim=0)
    # print("meand shape is ", accuracy.shape)
    # sys.exit()
    # print("dists shape is ",(dists.shape))
    # accuracy = torch.mean(dists, dim=0)
    # print("accuracy shape is ",(accuracy.shape))
    # return corrects
    # accuracy = corrects
    
    # print(corrects[0].item())
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
        # ret_acc[ite] = corrects[0].item()
    # ret_acc[0] = accuracy.item()
    # print(ret_acc)
    # sys.exit()
    return ret_acc

def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc
