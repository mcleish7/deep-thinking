import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import sys
import os 
from deepthinking import models as models
from deepthinking import utils as dt
# print(os.getcwd())
# os.chdir("test_time/")
# print(os.getcwd())
import numpy as np
import json
import matplotlib.pyplot as plt

os.chdir("/dcs/large/u2004277/deep-thinking/")
cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

# net = getattr(models, "dt_net_1d")(width=400, in_channels=3, max_iters=30) # for Verena
# state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-frockless-Verena/model_best.pth", map_location=device)

net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30) # for Betz => recall, alpha =0
state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida/model_best.pth", map_location=device)

# net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300) # for Jojo => recall, alpha =1
# state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo/model_best.pth", map_location=device)

net = net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(state_dict["net"])

net.eval()

def get_data():
    data = torch.load("batch_shells_sums/data/prefix_sums_data/48_data.pth").unsqueeze(1) - 0.5
    target = torch.load("batch_shells_sums/data/prefix_sums_data/48_targets.pth")
    input = data.to(device, dtype=torch.float) #to account for batching in real net
    target = target.to(device, dtype=torch.float)
    return input, target

def convert_to_bits(input): #moves from net output to one string of bits
    predicted = input.clone().argmax(1)
    golden_label = predicted.view(predicted.size(0), -1)
    return golden_label

def graph_progress(arr):
    plt.plot(arr)
    plt.title('Values of correct array')
    save_path = os.path.join("test_noise_outputs","test_noise_correctness.png")
    plt.savefig(save_path)

def net_out_to_bits(output,target, log = False, graph = False): #output from the net and the target bit string
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    best = output[:,bestind]
    if log == True:
        stats = correct.tolist()
        save_path = os.path.join("test_noise_outputs","test_noise_stats.json")
        with open(os.path.join(save_path), "w") as fp: #taken from train_model
            json.dump(stats, fp)
    if graph == True:
        graph_progress(correct)
    return convert_to_bits(best), correct[bestind] #returns the most accurate bit string and the number of bits which match with the target

batch_size = 500
channels = 1
width = 400
height = width
layer_types_input = [torch.nn.Conv1d]

class custom_func(fault_injection):
    count = 0
    j = 0 
    def __init__(self, in_j,model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.j = in_j

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400):
            j = self.j #between 0 and 48
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -20.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,j] = 20.0

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def count_to_correct(output,target): #output from the net and the target bit string
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct[50:])
    return bestind #returns the most accurate bit string and the number of bits which match with the target

def graph_time(arr1,arr2):
    plt.clf()
    plt.plot(arr1, linewidth = '3.0', label = "Recall")
    plt.plot(arr2, linewidth = '3.0', label = "Recall Prog")
    plt.title('Iterations to recover')
    plt.legend(loc="upper right")
    plt.yticks([0,26,5,10,25,15,20])
    save_path = os.path.join("test_time","test_time_correctness.png")
    plt.savefig(save_path)

print("now going into loop")
inputs,targets = get_data()
with torch.no_grad():
    time = []
    for index in range(0,10):
        average = []
        for i in range(0,inputs.size(0)):
            if (i%1000==0):
                print("starting iteration: ",i , " for index: ", index)
                file_path = os.path.join("test_time","time_tracker_2.txt")
                with open(file_path, 'w') as f:
                    f.write(f"starting iteration: {i} for index: {index}")
            input = inputs[i].unsqueeze(0)
            target = targets[i].unsqueeze(0)

            pfi_model_2 = custom_func(index,net, 
                                    batch_size,
                                    input_shape=[channels,width],
                                    layer_types=layer_types_input,
                                    use_cuda=True
                                )

            inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)
            inj_output = inj(input)
            average.append(count_to_correct(inj_output,target))
        mean = sum(average) / len(average)
        time.append(mean)
        file_path = os.path.join("test_time","time_list_tracker.txt")
        with open(file_path, 'r+') as f:
            f.write(f"for index: {index} the time array is {time}")
    print(time)
    graph_time(time)


# Betz data
#for index: 36 the time array is 
l = [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114]
# print("l is length ",len(l))
# for index: 39 the time array is  
l1 = [9.9924, 9.4904, 8.9798]
#redoing the first few:
#for index: 9 the time array is [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946]


# Jojo data
# for index: 39 the time array is 
j = [6.3722, 6.0566, 5.6513]
#for index: 36 the time array is 
j1 = [17.8872, 19.4591, 19.941, 19.912, 19.8543, 19.2299, 18.8148, 18.4023, 18.0458, 17.6382, 17.1686, 16.7601, 16.27, 15.8989, 15.4843, 15.1598, 14.6714, 14.3212, 13.8789, 13.4322, 13.014, 12.6209, 12.1961, 11.7285, 11.3529, 10.9505, 10.512, 10.1154, 9.707, 9.2398, 8.893, 8.5136, 8.1195, 7.723, 7.3973, 7.024, 6.6765]

betz = l+l1
jojo = j1+j
# graph_time(betz,jojo)