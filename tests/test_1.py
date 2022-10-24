# #reading in maze data
# import numpy as np
# data = np.load("test_shells_maze/data/maze_data_train_9/inputs.npy")
# lengths = [d for d in data]
# print(len(lengths))

# data = np.load("test_shells_maze/data/maze_data_test_33/inputs.npy")
# lengths = [d for d in data]
# print(len(lengths))

#reading in sums data
from re import T
import torch
# ex = torch.zeros((100, 1, 56), dtype=torch.float)
# #torch.Size([100, 1, 32]) = size of the inputs of 32 bit data into net
data = torch.load("data/prefix_sums_data/56_data.pth")[0]
# print(data)
print("in sums .pth files the input shape is "+str(data.shape))

# input = data.view(ex.size())
# print(input.shape)
# import torch
target = torch.load("data/prefix_sums_data/56_targets.pth")[0]
# print(target)
print("in sums .pth files the target shape is "+str(target.shape))

# from deepthinking.utils.prefix_sums_data import prepare_prefix_loader
# import tqdm
# data = prepare_prefix_loader(100, 500, 32, 56,train_split=0.8, shuffle=True)
# trainloader = data['train']
# for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
#     print("in trainloader the input shape is "+str(inputs.shape))
#     print("in trainloader the target shape is "+str(targets.shape))
#     exit

#reading maze data
import numpy as np
ex = np.zeros((100, 1, 56), dtype=float)
#torch.Size([100, 1, 32]) = size of the inputs of 32 bit data into net
# data = np.load("test_shells_maze/data/maze_data_test_33/inputs.npy")[0]
# # print(data)
# print("in mazes from the .npy files the data shape is "+str(data.shape))

# # input = data.view(ex.size())
# # print(input.shape)

# target = np.load("test_shells_maze/data/maze_data_train_9/inputs.npy")[0]
# # print(target)
# print("in mazes from the .npy files the target shape is "+str(target.shape))

print("!!!!!!!!from pycharm!!!!!!")
from easy_to_hard_data import PrefixSumDataset
import torch.utils.data as data
dataset = PrefixSumDataset("./data", num_bits=32)
# item = dataset.getitem(0)
it = iter(dataset)
item = next(it)
# print(item[0])
print("raw prefixsum download shape is " +str(item[0].shape))
input = item[0].unsqueeze(1) - 0.5
target = item[1].long()
# print(input)
print("input unsqueezed is "+str(input.shape))
# print(target)
print("target downloaded is " +str(target.shape))

trainloader = data.DataLoader(dataset, num_workers=0, batch_size=128,shuffle=False, drop_last=True)
it = iter(trainloader)
first = next(it)
print("loaders input shape is "+str(first[0].shape)) ##has 128 as has batched them together
print("loaders target shape is "+str(first[1].shape))

a = [[[ 0., -1.,  0., -1.,  0., -1., -1., -1.,  0., -1.,  0.,  0., -1.,  0.,-1., -1.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0., -1.,  0., -1.,  0.,-1., -1., -1., -1.]]]
t =[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1,1, 0, 0, 1, 1, 1, 1, 1]
# print(len(a[0][0]))
# print(len(t))
a = np.array(a)
a = torch.from_numpy(a)
t = np.array(t)
t = torch.from_numpy(t)
print(a.shape)
print(t.shape)