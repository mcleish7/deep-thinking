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

# net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Col => recall, alpha =0
# state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn/model_best.pth", map_location=device)

# net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Paden => recall, alpha =1
# state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-abased-Paden/model_best.pth", map_location=device)

# net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Lao => not recall, alpha =0
# state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao/model_best.pth", map_location=device)

def get_net():
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Paden => recall, alpha =1
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-abased-Paden/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_other_net():
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Col => recall, alpha =0
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

ex = torch.zeros((3, 1, 400), dtype=torch.float)

def get_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
# print("target unsqueezed is ",target.dtype)
# print(type(target))
# print("target dtype is ",target.dtype)
# print("target shape is ",target.shape)
input, target, a = get_data()
net = get_net()
# plot_maze(a.cpu(), t.cpu(), "maze_example.png") #plots the maze and solution
iters =300
corrects = torch.zeros(iters)

with torch.no_grad():
    output = net(input,iters_to_do=iters) 

print("output shape is " ,output.shape)
def convert_to_bits(output, input): #moves from net output to one string of bits
    predicted = output.clone().argmax(1)
    # print("predicted shape is ",predicted.shape)
    predicted = predicted.view(predicted.size(0), -1)
    # print("predicted shape 2 is ",predicted.shape)
    # if i == 25:
    #     with np.printoptions(threshold=np.inf):
    #         print(predicted.float().cpu().detach().numpy().tolist())
    #         print("\n\n")
    #         print((input.max(1)[0].view(input.size(0), -1)).cpu().detach().numpy().tolist())
    #     sys.exit()
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1))
    # print("gl shape is ",golden_label.shape)
    # print("extra bit shape is ",(input.max(1)[0].shape))
    return golden_label

def graph_progress(arr):
    plt.plot(arr*(100.0/1024.0), linewidth = '3.0', label = "dt_recall_prog")
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    save_path = os.path.join("test_noise_outputs","test_noise_correctness.png")
    plt.savefig(save_path)

def net_out_to_bits(input,output,target, log = False, graph = False): #output from the net and the target bit string
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]

        # print("outputi shape before is",outputi.shape)
        # torch.set_printoptions(profile="full")
        # print(outputi)
        golden_label = convert_to_bits(outputi, input)
        # print(golden_label)
        # torch.set_printoptions(profile="default") 
        # print("outputi shape after is",golden_label.shape)
        # print("before",target.dtype)
        target = target.view(target.size(0), -1)
        # print("after",target.dtype)
        # print("target shape is ",target.shape)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item()
        # if i ==50:
        #     np.save("50_maze_tensor",outputi.cpu().detach().numpy())
        # print(golden_label.dtype)
        # print(target.dtype)
        # print("corrects way is ",torch.amin(golden_label == target, dim=[0]).sum().item())
        # temp = torch.equal(golden_label,target)
        # print("new way is ",tf.reduce_sum(tf.cast(temp, tf.float32)))
        # res = golden_label.clone()
        # res[golden_label==0.0] = 1.0
        # res[golden_label==1.0] = 0.0
        # print(torch.equal(res, target))
        # sys.exit()
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    best = output[:,bestind]
    if log == True:
        stats = correct.tolist()
        save_path = os.path.join("test_noise_outputs","test_noise_stats.json")
        # with open(os.path.join(save_path), "w") as fp: #taken from train_model
        #     json.dump(stats, fp)
    if graph == True:
        graph_progress(correct)
    print("corrects is ",correct)
    print("corrects length is ",len(correct))
    return convert_to_bits(best, input), correct[bestind] #returns the most accurate bit string and the number of bits which match with the target

with torch.no_grad():
    t1,t2 = net_out_to_bits(input,output,target)
    # plot_maze(a.cpu(), t1.view((32,32)).cpu(), "maze_example_out.png")
    print("predicted string is ",t1)
    print("number of matches is ",t2)


batch_size = 1
channels = 3
width = 128
height = width
layer_types_input = ["all"]
# net = get_net()
# pfi_model = fault_injection(net, 
#                             batch_size,
#                             input_shape=[channels,width,height],
#                             layer_types=layer_types_input,
#                             use_cuda=True
#                             )
# print(pfi_model.print_pytorchfi_layer_summary())
# b = [0]
# layer = [4]
# C = [3]
# H = [10]
# W = [10]
# err_val = [1]
# inj = pfi_model.declare_neuron_fi(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val) #dim 3 ignored for sums
# input, target, a = get_data()
# inj_output = inj(input)

# inj_label,matches = net_out_to_bits(input,inj_output,target)

# print("[Single Error] PytorchFI label:", inj_label)
# print("matches in inj_label is ",matches)

class custom_func(fault_injection):
    # count = 0
    def __init__(self, input, input_change, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.input = input
        self.input_to_change_to = input_change

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        layer_from =25
        layer_to = 27
        # file_path = os.path.join("test_time","time_list_tracker.txt")
        with open("test_change_ends.txt", 'a+') as f:
            f.write(f"output shape is {output.shape} at {self.get_current_layer()}\n")
        # print("output shape is ",output.shape," at ",self.get_current_layer())
        # print("input shape is ",output.shape," at ",self.get_current_layer())
        # if (self.get_current_layer() >= (layer_from*7)) and (self.get_current_layer() <= ((layer_to*7)+1)):
            # print("output shape is ",output.shape," at ",self.get_current_layer())
            # print(self.get_current_layer())
            # output[:] = torch.zeros(output.shape)
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

net = get_net()

with torch.no_grad():
    input_change = None #TODO
    pfi_model_2 = custom_func(input, input_change, net, 
                            batch_size,
                            input_shape=[channels,width,height],
                            layer_types=layer_types_input,
                            use_cuda=True
                        )
    # print(pfi_model_2.print_pytorchfi_layer_summary())
    inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)

    inj_output = inj(input)

    inj_label,matches = net_out_to_bits(input,inj_output,target,graph=False)
    # , log=True, graph=True)
    print("[Single Error] PytorchFI label from class:", inj_label)
    print("matches in inj_label is ",matches)

# other_net = get_other_net()
# with torch.no_grad():
#     pfi_model_3 = custom_func(other_net, 
#                             batch_size,
#                             input_shape=[channels,width,height],
#                             layer_types=layer_types_input,
#                             use_cuda=True
#                         )
#     # print(pfi_model_2.print_pytorchfi_layer_summary())
#     inj = pfi_model_3.declare_neuron_fi(function=pfi_model_2.flip_all)

#     inj_output_2 = inj(input)

#     inj_label,matches = net_out_to_bits(input,inj_output_2,target,graph=False)
#     # , log=True, graph=True)
#     print("[Single Error] PytorchFI label from class:", inj_label)
#     print("matches in inj_label is ",matches)

def graph_two_helper(output,input,target):
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]

        golden_label = convert_to_bits(outputi, input)

        target = target.view(target.size(0), -1)

        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item()
    correct = corrects.cpu().detach().numpy()
    return correct

def graph_two(input,output_1,output_2,target): #output from the net and the target bit string
    data_1 = graph_two_helper(output_1,input,target)
    data_2 = graph_two_helper(output_2,input,target)
    plt.plot(data_1*(100.0/1024.0), linewidth = '3.0', label = "dt_recall_prog")
    plt.plot(data_2*(100.0/1024.0), linewidth = '3.0', label = "dt_recall")
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    save_path = os.path.join("test_noise_outputs","repo_7.png")
    plt.savefig(save_path)

# graph_two(input,inj_output,inj_output_2,target)
