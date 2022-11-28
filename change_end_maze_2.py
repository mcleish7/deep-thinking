import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import deepthinking.models as models
import deepthinking.utils as dt
import numpy as np
import sys
import os
import json
import re
# import tensorflow as tf
import matplotlib.pyplot as plt
from easy_to_hard_plot import plot_maze
from easy_to_hard_plot import MazeDataset
import random

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

# net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Lao => not recall, alpha =0
# state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao/model_best.pth", map_location=device)

def get_net():
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50) # for Paden => recall, alpha =1
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
    # print(predicted.get_device())
    # print(input.get_device())
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

# with torch.no_grad():
    # t1,t2 = net_out_to_bits(input,output,target)
    # plot_maze(a.cpu(), t1.view((32,32)).cpu(), "maze_example_out.png")
    # print("predicted string is ",t1)
    # print("number of matches is ",t2)


# given in input and target we move the begininng and end one step further apart
def move_back_2(input,target):
    # input is of shape [3,32,32]
    # target is of shape [32,32]
    print(input.shape)
    print(target.shape)
    plot_maze(input,target.cpu().numpy(), "change_ends_dev")

def move_end(input,target):
    #move end of the maze to a random location that is NOT next to the start

    # input[0] is the maze, no start, with end --do nothing
    # input[1] is the maze, with start, no end --need to fill in old endpoint and remove new one
    # input[2] is the maze, no start or end --need to fill in old endpoint and remove new one
    # so can use middle to fill in 0 and 2 then take it out of the other two
    print(input.shape)
    print(target.shape)
    i = random.randint(0,30) #only up to 30 so can always take next one to be in picture too
    j = random.randint(0,30)
    print("i,j is ",input[0][i][j].cpu().tolist())
    # change = 0 if input.cpu().tolist()[0][i][j] == 0.0 else 1
    temp0 = input[0].clone()
    temp1 = input[1]
    temp2 = input[2]
    # input[0] = temp0
    # input[1] = temp0
    # input[2] = temp0
    # plot_maze(input,target.cpu().numpy(), "change_ends_dev_1")

    while input.cpu().tolist()[0][i][j] != 1:
        i = random.randint(0,30)
        j = random.randint(0,30)
    for a in range(0,3):
        for y in range(0,2):
            # print("x is ",input.cpu().tolist()[a][i+x][j])
            if input.cpu().tolist()[a][i][j+y] == 0.0: #need to know to go left or right
                y = y * -1
            for x in range(0,2):
                if input.cpu().tolist()[a][i+x][j] == 0.0: #need to know to go left or right
                    x = x * -1
                print("painting black square which was ",input.cpu().tolist()[a][i+x][j+y])
                input[a][i+x][j+y] = 0

    t1 = torch.add(input[0], input[1]).bool().int() #fills in old end
    t2 = torch.add(input[2], input[0]).bool().int() #fills in old end

    input[0] = temp0
    input[1] = t1
    input[2] = t2

    # input[0] = temp0
    # input[1] = temp0
    # input[2] = temp0
    plot_maze(input,target.cpu().numpy(), "change_ends_dev")

def move_end_on_target_path(input,target):
    #move end of the maze to a random location that is on target path

    # input[0] is the maze, no start, with end --do nothing
    # input[1] is the maze, with start, no end --need to fill in old endpoint and remove new one
    # input[2] is the maze, no start or end --need to fill in old endpoint and remove new one
    # so can use middle to fill in 0 and 2 then take it out of the other two

    i = random.randint(0,30) #only up to 30 so can always take next one to be in picture too
    j = random.randint(0,30)
    print("tagrte type is ",type(target))
    temp0 = input[0].clone()

    while target.cpu().tolist()[i][j] != 1:
        i = (random.randint(0,15))*2
        j = (random.randint(0,15))*2 #so always evem
    for a in range(0,3):
        for y in range(0,2):
            # print("x is ",input.cpu().tolist()[a][i+x][j])
            if target.cpu().tolist()[i][j+y] == 0.0: #need to know to go left or right
                y = y * -1
            for x in range(0,2):
                if target.cpu().tolist()[i+x][j] == 0.0: #need to know to go left or right
                    x = x * -1
                print("painting black square which was ",input.cpu().tolist()[a][i+x][j+y])
                input[a][i+x][j+y] = 0

    t1 = torch.add(input[0], input[1]).bool().int() #fills in old end
    t2 = torch.add(input[2], input[0]).bool().int() #fills in old end

    input[0] = temp0
    input[1] = t1
    input[2] = t2
    # np.save("change_end_dev",input.cpu().numpy())
    plot_maze(input,target.cpu().numpy(), "change_ends_dev")

input = torch.from_numpy(np.load("change_end_dev.npy"))
# move_end_on_target_path(input[0],target[0])
# print("input shape is",input.shape)
target = target[0] 
for i in range(0,32):#manually creating a new target for now
    for j in range(0,11):
        target[i][j] = 0
# print(target.shape)
# print(input[1][27])
# print(target[27])
plot_maze(input,target.cpu().numpy(), "change_ends_dev")
np.save("change_end_dev_input",input.cpu().numpy())
np.save("change_end_dev_target",input.cpu().numpy())

input_after_50 = input
target_after_50 = target

#run for 50 iteration on orgional data
store = []
class custom_func(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        # each recurrent block is length 8, with 5 128's then 32,8,2
        layer_from = 49
        layer_to = 50
        if (self.get_current_layer()>(layer_from*8)) and (self.get_current_layer()<=(layer_to*8)):
            # print("at layer ",self.get_current_layer()," size is ",output.shape)
            global store
            store.append(output)
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

def tester(net):
    batch_size = 1
    channels = 3
    width = 32
    height = width
    layer_types_input = [torch.nn.Conv2d]
    with torch.no_grad():
        pfi_model = custom_func(net, 
                                batch_size,
                                input_shape=[channels,width,height],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)

        return inj(input)
net = get_net()
input, target, a = get_data()
input = input[0].unsqueeze(0)
begin_input = input
target = target[0].unsqueeze(0)
input= input.to(device)
target =target.to(device)
with torch.no_grad():
    output_first_50 = tester(net)

t1,t2 = net_out_to_bits(input,output_first_50,target)
print("corrects is ",t1)
print("nuber of corrects is ",t2)

#run for 50 iterations on altered data but for first epoch change in pytorchfi to be last epoch of orgional 50
#store contains what we want to overwrite with
class custom_func_2(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        # each recurrent block is length 8, with 5 128's then 32,8,2
        layer_from = 0
        layer_to = 2
        if (self.get_current_layer()>(layer_from*8)) and (self.get_current_layer()<=(layer_to*8)):
            # print("before",output)
            # print("at layer ",self.get_current_layer()," size is ",output.shape)
            global store
            # print(store[self.get_current_layer()-1-(8*(layer_to-1))].shape)
            output = store[self.get_current_layer()-1-(8*(layer_to-1))]
            # print("after",output)
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

def tester_2(net):
    batch_size = 1
    channels = 3
    width = 32
    height = width
    layer_types_input = [torch.nn.Conv2d]
    with torch.no_grad():
        pfi_model = custom_func_2(net, 
                                batch_size,
                                input_shape=[channels,width,height],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)

        return inj(input)

del net

net = get_other_net()
input, target = input_after_50, target_after_50
input = input.unsqueeze(0)
target = target.unsqueeze(0)
input= input.to(device)
target =target.to(device)
with torch.no_grad():
    output_final= tester_2(net)

t1,t2 = net_out_to_bits(input,output_final,target)
print("corrects is after second 50 ",t1)
print("nuber of corrects is after second 50",t2)

def print_end_and_beg(output1,input1, output2,input2):
    label_1 = convert_to_bits(output1[:, 50], input1).reshape((32,32))
    input2=input2.unsqueeze(0).to(device)
    # print(output2.shape)
    # print(input2.shape)
    # label_2 = convert_to_bits(output2[:, 0], input2).reshape((32,32))
    # print(label_2.shape)
    fig, axs = plt.subplots(4, 5, figsize=(10, 5)) 
    # axs[0].imshow(label_1.cpu(),cmap='Greys_r') #this is the output of the 50th iteration of run 1
    for i in range(0,4):
        for j in range(0,5):
            cur = (i*5)+j
            label = convert_to_bits(output2[:, cur], input2).reshape((32,32))
            axs[i][j].imshow(label.cpu(),cmap='Greys_r')
    # axs[1].imshow(label_2.cpu(),cmap='Greys_r')
    # label_3 = convert_to_bits(output2[:, 1], input2).reshape((32,32))
    # axs[2].imshow(label_3.cpu(),cmap='Greys_r')
    # label_4 = convert_to_bits(output2[:, 2], input2).reshape((32,32))
    # axs[3].imshow(label_4.cpu(),cmap='Greys_r')
    plt.savefig("plot_start_after_50_change_end", bbox_inches="tight")
    plt.close()

print_end_and_beg(output_first_50,begin_input,output_final,input_after_50)


#attempt 3now trying by wrriting the output of run1 to only the end of the recurrent moduel at the end of epoch 1 and 2
class custom_func_3(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        # each recurrent block is length 8, with 5 128's then 32,8,2
        # print("at layer ",self.get_current_layer()," size is ",output.shape)
        layer_from = 0
        layer_to = 2
        if (self.get_current_layer()>(layer_from*8)) and (self.get_current_layer()<=(layer_to*8)):
            # print("before",output)
            if output.size(1)==2:
                global output_first_50
                # print(output_first_50.squeeze()[int(self.get_current_layer()/8)-1:int(self.get_current_layer()/8)].shape)
                # print(int(self.get_current_layer()/16)+50)
                output = output_first_50.squeeze()[int(self.get_current_layer()/8)-1+50:int(self.get_current_layer()/8)+50]
            # print("after",output)
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

def tester_3(net):
    batch_size = 1
    channels = 3
    width = 32
    height = width
    layer_types_input = ["all"]
    with torch.no_grad():
        pfi_model = custom_func_3(net, 
                                batch_size,
                                input_shape=[channels,width,height],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)

        return inj(input)

net = get_other_net()
input, target = input_after_50, target_after_50
input = input.unsqueeze(0)
target = target.unsqueeze(0)
input= input.to(device)
target =target.to(device)
with torch.no_grad():
    output_final= tester_3(net)

t1,t2 = net_out_to_bits(input,output_final,target)
print("corrects is after second 50 ",t1)
print("nuber of corrects is after second 50",t2)

def print_20(output2,input2):
    input2=input2.unsqueeze(0).to(device)
    fig, axs = plt.subplots(4, 5, figsize=(10, 5)) 
    # axs[0].imshow(label_1.cpu(),cmap='Greys_r') #this is the output of the 50th iteration of run 1
    for i in range(0,4):
        for j in range(0,5):
            cur = (i*5)+j
            label = convert_to_bits(output2[:, cur], input2).reshape((32,32))
            axs[i][j].imshow(label.cpu(),cmap='Greys_r')
    # label = convert_to_bits(output_first_50.squeeze()[int(8/8)+50-1:int(8/8)+50], input2).reshape((32,32))
    # axs[0][0].imshow(label.cpu(),cmap='Greys_r')
    plt.savefig("plot_start_after_50_change_end", bbox_inches="tight")
    plt.close()

print_20(output_final,input_after_50)

#forth try is to input the output as the input of the next maze
net = get_net()
input, target = input_after_50, target_after_50
input= input.to(device)
target =target.to(device)
# print(target.shape)
# print(output_first_50.squeeze()[50].shape)
golden_label_after_50 = convert_to_bits(output_first_50.squeeze()[50].unsqueeze(0),begin_input).reshape((32,32))
# print(golden_label_after_50.shape)
# input[0] is the maze, no start, with end 
# input[1] is the maze, with start, no end 
# input[2] is the maze, no start or end 

end_only  = input[0].clone()-input[2].clone() #is just the end
start_only = input[1].clone()-input[2].clone()
input[2] = torch.mul(input[2].clone().flatten(),golden_label_after_50.clone().flatten()).reshape((32,32))
input[0] = input[2].clone() + end_only
input[1] = input[2].clone() + start_only

# fig, axs = plt.subplots(figsize=(10, 5))
# axs.imshow(input[0].cpu(),cmap='Greys_r')
# plt.savefig("plot_start_after_50_change_end", bbox_inches="tight")
# plt.close()

input = input.unsqueeze(0)
target = target.unsqueeze(0)
with torch.no_grad():
    output_final= net(input,max_iters = 100)

t1,t2 = net_out_to_bits(input,output_final,target)
print("corrects is after second 50 ",t1)
print("nuber of corrects is after second 50",t2)