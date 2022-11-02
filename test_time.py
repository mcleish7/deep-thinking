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
                    output[0,0,j] = -50.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,j] = 50.0

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

def graph_time(arr):
    plt.clf()
    plt.plot(arr)
    plt.title('Iterations to recover')
    save_path = os.path.join("test_time","test_time_correctness.png")
    plt.savefig(save_path)

print("now going into loop")
inputs,targets = get_data()
with torch.no_grad():
    time = []
    for index in range(0,40):
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

