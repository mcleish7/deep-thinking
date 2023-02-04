import os,torch,sys
import numpy as np
folder_name = "data/graph_train_6"
inputs_path = os.path.join(folder_name, "inputs.npz")
solutions_path = os.path.join(folder_name, "solutions.npz")
inputs_np = np.load(inputs_path)['arr_0']
targets_np = np.load(solutions_path)['arr_0']

# print(inputs_np.shape)
# print(inputs_np[0])

# print(targets_np.shape)
# print(targets_np[10])


# size  = len(inputs_np.files)
# index = inputs_np.files
# arr = inputs_np[index[0]]
# for i in range(1,size):
#     arr = np.vstack((arr,inputs_np[index[i]]))
# print(arr.shape)

# for i in range(1, nI):
#     print(i)
    # np.vstack()
# inputs = torch.from_numpy(inputs_np).float()
# targets = torch.from_numpy(targets_np).long()

# arr = np.random.rand(6,6)
# arr2 = np.random.rand(6,6)
# arr3 = np.append([arr],[arr2], axis=0)
# print(arr3.shape)
# print(arr3)

# import dm-clrs as clrs
import clrs
help()
help()
help(clrs)
