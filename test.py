# test for checking cuda output
# import torch
# import tensorflow
# print("running")
# print(torch.cuda.is_available())


# reading .pth objects which are pickled, are actually just matracies
# import torch
# obj = torch.load('testing_shells/data/prefix_sums_data/16_data.pth')
# print(obj)

# prints the number of cuda cores i.e. the number of cuda capable GPUs
# import torch
# print(torch.cuda.device_count())
# cur = torch.cuda.current_device()
# print(torch.cuda.get_device_name(cur)) #prints the name of the GPU in the device

# python3.9 /dcs/large/u2004277/deep-thinking/test_model.py problem.model.model_path=../../../../batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida problem=mazes problem.model.test_iterations.low=0 problem.model.test_iterations.high=10

import torch 
# inp = torch.rand([1,2,32])
# def convert_to_bits(input): #moves from net output to one string of bits
#     print("in shape is ", input.shape)
#     print("in is ", input)
#     predicted = input.clone().argmax(1)
#     print("predicted shape is ",predicted.shape)
#     print("predicted is ",predicted)
#     golden_label = predicted.view(predicted.size(0), -1)
#     print("golden label shape is ", golden_label.shape)
#     print("golden label is ", golden_label)
#     return golden_label
# print(inp)  
# # convert_to_bits(inp)
# j = 0
# if inp[0,0,j] > 0.5:
#     inp[0,0,j] = 1.0
#     inp[0,1,j] = 1.0
# else:
#     inp[0,0,j] = 0.0
#     inp[0,1,j] = 0.0
# print(inp)


# import sys
# device = "cuda"
# def get_data():
#     data = torch.load("../batch_shells_sums/data/prefix_sums_data/48_data.pth").unsqueeze(1) - 0.5
#     target = torch.load("../batch_shells_sums/data/prefix_sums_data/48_targets.pth")
#     # print(a.shape)
#     input = data.to(device, dtype=torch.float) #to account for batching in real net
#     # print(input.shape)
#     # print(input)
#     target = target.to(device, dtype=torch.float)#.long()
#     return input, target

# inputs, targets = get_data()

# print(inputs.shape)
# for i in range(0,inputs.size(0)):
#     input = inputs[i].unsqueeze(0)
#     print(input.shape)
#     print(input)
#     target = targets[i].unsqueeze(0)
#     print(target.shape)
#     sys.exit()



# import numpy as np
# import sys
# inp = torch.rand([1, 50, 2, 32, 32])

# def l2_norm(output): #output from net
#     # inputs will be size: [1, 50, 2, 32, 32] to be split into [1, 2, 32, 32]
#     out = []
#     output1 = output[:, 0]
#     output1 = output1.cpu().detach().numpy().flatten()
#     for i in range(0,output.size(1)-1):
#         output2 = output[:, i+1]
#         output2 = output2.cpu().detach().numpy().flatten()
#         norm = np.sum(np.power((output1-output2),2))
#         out.append(norm)
#         output1= output2
#     return out

# print(l2_norm(inp))

# import os
# print(os.getcwd())
# os.chdir("/dcs/large/u2004277/deep-thinking/")
# print(os.getcwd())

# import os
# import shutil

# from_folder = "graph_6"
# to_folder = "data"
# dir = os.path.join("~/Desktop/graph_generation_files", from_folder)
# dir = os.path.expanduser(dir)
# path = os.path.join(to_folder, from_folder)
# shutil.copytree(dir, path)

from deepthinking.utils.graph_data import prepare_graph_loader
# prepare_graph_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True)
prepare_graph_loader(100, 100, 6, 6, False)

