import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import deepthinking.models as models
import deepthinking.utils as dt
import numpy as np
import sys
import os
import json
# import tensorflow as tf
import matplotlib.pyplot as plt
from easy_to_hard_plot import plot_maze
from easy_to_hard_plot import MazeDataset

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_no_prog_net():
    net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Lao => not recall, alpha =0
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_prog_net():
    net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Cor => not recall, alpha =1
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-distinct-Cornesha/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_recall_prog_net():
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Paden => recall, alpha =1
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-abased-Paden/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_recall_no_prog_net():
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Col => recall, alpha =0
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

ex = torch.zeros((3, 1, 400), dtype=torch.float)

def get_data():
    data = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")
    target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")
    # data = np.load("batch_reproduce_5/data/maze_data_test_33/inputs.npy")
    # target = np.load("batch_reproduce_5/data/maze_data_test_33/solutions.npy")
    a = data[1]
    a = torch.from_numpy(a)
    input = a.to(device, dtype=torch.float).unsqueeze(0) #to account for batching in real net
    print("input shape is ",input.shape)

    b = target[1]
    t = torch.from_numpy(b)
    print("t is ",t.dtype)
    t = t.to(device, dtype=torch.float)#.long()
    print("t in middle is ",t.dtype)
    target = t.unsqueeze(0)
    return input, target, a

input, target, a = get_data()

def convert_to_bits(output, input): #moves from net output to one string of bits
    predicted = output.clone().argmax(1)
    # print(predicted.shape)
    predicted = predicted.view(predicted.size(0), -1)
    # print(predicted.shape)
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1))
    # print(golden_label.shape)
    return golden_label

def l2_norm(output): #output from net
    # inputs will be size: [1, 50, 2, 32, 32] to be split into [1, 2, 32, 32]
    # print("input to norm function shape is ", output.shape)
    out = []
    output1 = output[:, 0]
    output1 = output1.cpu().detach().numpy().flatten()
    for i in range(0,output.size(1)-1):
        output2 = output[:, i+1]
        output2 = output2.cpu().detach().numpy().flatten()
        norm = np.sum(np.power((output1-output2),2))
        out.append(norm)
        output1= output2
    return out

def net_out_to_bits(input,output,target, log = False, graph = False): #output from the net and the target bit string
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi, input)
        target = target.view(target.size(0), -1)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item()
        if i ==50:
            np.save("50_maze_tensor",outputi.cpu().detach().numpy())
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    best = output[:,bestind]
    # print("corrects is ",correct)
    # print("corrects length is ",len(correct))
    return convert_to_bits(best, input), correct[bestind] #returns the most accurate bit string and the number of bits which match with the target

batch_size = 1
channels = 3
width = 128
height = width
layer_types_input = [torch.nn.Conv2d]

class custom_func(fault_injection):
    # count = 0
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

def tester(net):
    with torch.no_grad():
        pfi_model = custom_func(net, 
                                batch_size,
                                input_shape=[channels,width,height],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)

        return inj(input)

recall_prog_output = tester(get_recall_prog_net())
recall_no_prog_output = tester(get_recall_no_prog_net())
prog_output = tester(get_prog_net())
no_prog_output = tester(get_no_prog_net())

def graph_norm_progress(arr1, arr2, arr3, arr4):
    plt.clf()
    plt.plot(arr1, linewidth = '3.0', label = "dt_recall_prog")
    plt.plot(arr2, linewidth = '3.0', label = "dt_recall")
    plt.plot(arr3, linewidth = '3.0', label = "dt_prog")
    plt.plot(arr4, linewidth = '3.0', label = "dt")
    plt.title('Change in features over time')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Δφ')
    plt.legend(loc="upper right")
    save_path = os.path.join("test_noise_outputs","test_changes_correctness.png")
    plt.savefig(save_path)

# graph_two(input,inj_output,inj_output_2,target)
recall_prog_norm = l2_norm(recall_prog_output)
recall_no_prog_norm = l2_norm(recall_no_prog_output)
prog_norm = l2_norm(prog_output)
no_prog_norm = l2_norm(no_prog_output)

graph_norm_progress(recall_prog_norm, recall_no_prog_norm, prog_norm, no_prog_norm)