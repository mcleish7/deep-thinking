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
import numpy as np
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

# prepare_graph_loader(100, 100, 6, 6, False)

# inputs_path = "tests/data/graph_test_6/inputs.npz"
# inputs_np = np.load(inputs_path)['arr_0']
# print(inputs_np.shape)
# print(inputs_np[0])
# print(np.any(np.isnan(inputs_np)))

# sols_path = "tests/data/graph_test_6/solutions.npz"
# sols_np = np.load(sols_path)['arr_0']
# print(sols_np.shape)
# print(sols_np[0])


def get_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")
    # target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")
    data = np.load("batch_reproduce_5/data/maze_data_test_33/inputs.npy")
    target = np.load("batch_reproduce_5/data/maze_data_test_33/solutions.npy")
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

# i,t,a = get_data()
# print(i.shape) #torch.Size([1, 3, 32, 32]) for 13x13, for 33x33 32 goes to 72
# print(t.shape) #torch.Size([1, 32, 32]) for 13x13
# current error
# RuntimeError: Given groups=1, weight of size [128, 3, 3, 3], expected input[50, 1, 6, 6] to have 3 channels, but got 1 channels instead
# changed in_channels to 1 and new error is 
# RuntimeError: Expected target size [50, 36], got [50, 1]
import matplotlib.pyplot as plt
import os
# Betz data
l = [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114]
# print("l is length ",len(l))
# for index: 39 the time array is  
l1 = [9.9924, 9.4904, 8.9798]
l = l+l1
# print("l is length ",len(l))
# print(l)

# [18, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114]
def graph_progress(arr, title, folder, file):
    plt.plot(arr)
    plt.title(title)
    save_path = os.path.join(folder,file)
    plt.savefig(save_path)

# graph_progress(l,'Time to recover from pertubation',"test_time","test_time_Betz_correctness.png")

arr = [[[ 0, -1, -1, -1, -1, -1],
  [-1,  0,  1,  1,  2,  1],
  [-1,  1,  0,  2,  1,  1,],
  [-1,  1,  2,  0,  3,  2],
  [-1,  2,  1,  3,  0,  2],
  [-1,  1,  1,  2,  2,  0]],
 [[ 0, -1,  1,  1, -1,  2],
  [-1,  0, -1, -1,  1, -1],
  [ 1, -1,  0,  1, -1,  1],
  [ 1, -1,  1,  0, -1,  1],
  [-1,  1, -1, -1,  0, -1],
  [ 2, -1,  1,  1, -1,  0]]]
# X = np.array(arr)
# print(X.shape)
# XNormed = (X - X.mean())/(X.std())
# print(XNormed)

# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format('graph_generation/graph_generation_files/graph_5/test_filename.emb')
# print(type(model))
# print(model)
# weights = torch.FloatTensor(model.vectors)
# print(weights)
# first=weights
# second = torch.load()
# print(torch.eq(first, second))


from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(arr[0],arr[1])
# print(round(mse, 2))

c = (np.array(arr) >-1).astype(int)
# print(c)
input = torch.from_numpy(np.array([[ 0, 2, -1, -1, -1, -1],
  [-1,  0,  1,  1,  2,  1],
  [-1,  1,  0,  2,  1,  1,],
  [-1,  1,  2,  0,  3,  2],
  [-1,  2,  1,  3,  0,  2],
  [-1,  1,  1,  2,  2,  0]]))
# input = torch.zeros([6,6])
predicted = torch.rand(6,6)
golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1))
other = (input.max(1)[0].view(input.size(0), -1))
# print(input)
# print(predicted)
# print(other)
# print(golden_label)


def get_data_maze():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")
    target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")
    # data = np.load("batch_reproduce_5/data/maze_data_test_33/inputs.npy")
    # target = np.load("batch_reproduce_5/data/maze_data_test_33/solutions.npy")
    a = data[1]
    a = torch.from_numpy(a)
    input = a.to(device, dtype=torch.float).unsqueeze(0) #to account for batching in real net
    # print("input shape is ",input.shape)

    b = target[1]
    t = torch.from_numpy(b)
    # print("t is ",t.dtype)
    t = t.to(device, dtype=torch.float)#.long()
    # print("t in middle is ",t.dtype)
    target = t.unsqueeze(0)
    return input, target, a


# from easy_to_hard_plot import plot_maze --- copied code in
import seaborn as sns
def plot_maze(inputs, targets, save_str):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    ax = axs[0]
    print("permutes shape is ",inputs.cpu().squeeze().permute(1, 2, 0).shape)
    ax.imshow(inputs.cpu().squeeze().permute(1, 2, 0))
    #removes the axis for next 4 lines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    ax = axs[1]
    sns.heatmap(targets, ax=ax, cbar=False, linewidths=0, square=True, rasterized=True)
    #removes the axis for next 4 lines
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(save_str, bbox_inches="tight")
    plt.close()
  
# input, target, a = get_data_maze()
# inputnp = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")[1]
# target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")[1]
# print("input is ",input.shape)
# print("target is ",target.shape)
# with np.printoptions(threshold=np.inf):
#   torch.set_printoptions(profile="full")
#   # print(input[0,1])
#   torch.set_printoptions(profile="default")

# fig, axs = plt.subplots(1, 5, figsize=(10, 5))
# axs[0].imshow(input.cpu().squeeze().permute(1, 2, 0))
# sns.heatmap(target, ax=axs[1], cbar=False, linewidths=0, square=True, rasterized=True)
# print("shape is ",input.cpu().squeeze().permute(1, 2, 0).shape)
# print("shape no perm is ",input.cpu().squeeze().shape)
# print("shape no perm 0 is ",input.cpu().squeeze()[0].shape)
# axs[2].imshow(input.cpu().squeeze()[0],cmap='Greys')
# axs[3].imshow(input.cpu().squeeze()[1],cmap='Greys')
# axs[4].imshow(input.cpu().squeeze()[2],cmap='Greys')
# plt.tight_layout()
# plt.savefig("test_1", bbox_inches="tight")
# plt.close()

# plot_maze(input,target,"testing")

# c = np.load("compare_time.npy")
# c = np.transpose(c)[:35]
# print(c.shape)
# ax = sns.heatmap(c, linewidth=0.5)
# ax.invert_yaxis()
# ax.set(xlabel='index of bit flipped', ylabel='iteration after peturbation', title="Average number of bits different")
# plt.savefig("test_heat", bbox_inches="tight")

# c = np.load("test_noise_outputs/test_maze_mismatch_averages.npy")
# print(c.shape)
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
import torchvision
import sys
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
# print(type(trainset))
# i,j = trainset[1]
# print(type(i))
# print(type(j))
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
# for i, data in enumerate(trainloader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     inputs, labels = data
#     print("input type is ", inputs.shape)
#     print("label type is ",labels.shape)
#     sys.exit()
preds = torch.rand(500,1)
im = torch.rand(500,1)

im_loss = torch.pow(preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1).mean()
loss = im_loss.mean()
# print("preds is ",preds)
# print("im is ",im)
# print("shape preds[:, -1:] is ",preds[:, -1:]) # takes the last column of preds i.e. the last ouput of the network when they are all batches together
# print("shape im[:, None, :] is ",im[:, None, :]) # puts each row into another array so it is an array of an array
# # print("minus is ",(preds[:, -1:] - im[:, None, :])) #is shape [5,5,5], takes each row of im and subracts the value of the coumn of im, giving a [5,5] then for each row is 5 times
# print("minus is ",(preds[:, -1:] - im[:, None, :]).mean(dim=-1)) # shape [5,5], takes mean along each row
# print("shape of loss is ", im_loss) # [5], takes mean along each row again

# print("shape preds[:, -1:] is ",preds[:, -1:].shape) # takes the last column of preds i.e. the last ouput of the network when they are all batches together
# print("shape im[:, None, :] is ",im[:, None, :].shape) 
# print("shape of loss im is ", im_loss) # [5], takes mean along each row again
# print("shape of loss is ",loss)

def graph_maze_mismatch():
    arr1= np.load("AvonTests/noise_maze_array_part_1.npy")
    arr2 = np.load("AvonTests/noise_maze_array.npy")
    arr = np.concatenate((arr1,arr2))
    # arr = arr.T
    plt.clf()
    alphas = ["0.01","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
    denom = 15376.0
    # print(arr.shape[0])
    # sys.exit()
    for i in range(0,arr.shape[0]):
        run = arr[i]
        # print("run is type",type(run))
        # print(run.shape)
        plt.plot(run*(100.0/denom), linewidth = '1.5', label = alphas[i])
    plt.title('Accuracy over time when features are swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    # file_name = f"maze_mismatch.png"
    file_name = "rough.png"
    save_path = os.path.join("test_noise_outputs",file_name)
    plt.savefig(save_path, dpi=500)
# print(arr.shape)
# plt.plot(arr.T)
# plt.savefig("rough", bbox_inches="tight")
graph_maze_mismatch()