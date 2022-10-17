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
import torch
print(torch.cuda.device_count())
cur = torch.cuda.current_device()
print(torch.cuda.get_device_name(cur)) #prints the name of the GPU in the device 