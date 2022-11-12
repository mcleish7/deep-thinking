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

# net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30) # for Betz => recall, alpha =0
# state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida/model_best.pth", map_location=device)

net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300) # for Jojo => recall, alpha =1
state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo/model_best.pth", map_location=device)

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

def first_diff(arr1,arr2):
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return i
    return -1

def same(arr1,arr2):
    if len(arr1)!=len(arr2):
        return False
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True

def num_diff(arr1,arr2):
    count = 0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1
    return count

def track_after_peturb(output,target):
    target = target.cpu().detach().numpy().astype(int)[0]
    # print(target)
    out = []
    change = []
    first_difference = []
    num_diffences = []
    correct = []
    # print("output shape is",output.shape)
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        # print(golden_label.cpu().detach().numpy()[0])
        if i == 50:
            out.append(golden_label.cpu().detach().numpy().astype(int)[0])
        if (i>50) and (i<100):
            outi=i-50
            # print(outi-1)
            # print("triggered")
            # print(golden_label)
            gl = golden_label.cpu().detach().numpy().astype(int)[0]
            out.append(gl)
            prev = out[outi-1]
            match = same(gl,target)
            # correct = "true" if match else "false"
            # change.append(str(outi)+", "+str(first_diff(gl,prev))+", "+str(num_diff(gl,prev))+", "+correct) # string (index after 50, index of first difference, number of differences, if the output is correct)
            first_difference.append(first_diff(gl,prev))
            num_diffences.append(num_diff(gl,prev))
            correct.append(match)
    return first_difference, num_diffences, correct
    
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

def density(num_diffences, correct, input, target):
    count = 0
    total = 0
    for i in range(len(correct)):
        if correct[i] == False:
            count +=1
            total += num_diffences[i]
    if count >0:
        return total/count
    else:
        #when this prints it is a very rare case where the net recovers in one step
        print("one step recovery")
        print(input)
        print(target)
        print(num_diffences)
        print(correct)
        return 0

print("now going into loop")
inputs,targets = get_data()
with torch.no_grad():
    # time = []
    average_density = []
    for index in range(0,40):
    # for index in range(0,10):
        # average = []
        density_list = []
        for i in range(0,inputs.size(0)):
        # for i in range(0,1000):
            # if (i%1000==0):
                # print("starting iteration: ",i , " for index: ", index)
                # file_path = os.path.join("test_time","time_tracker_2.txt")
                # with open(file_path, 'w') as f:
                #     f.write(f"starting iteration: {i} for index: {index}")
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
            # average.append(count_to_correct(inj_output,target))
            first_difference, num_diffences, correct = track_after_peturb(inj_output,target)
            density_list.append(density(num_diffences, correct, input, target))
        mean = sum(density_list) / len(density_list)
        # print(mean)
        average_density.append(mean)
        print("average density list is ",average_density)
        file_path = os.path.join("test_time","time_track_out.txt")
        with open(file_path, 'r+') as f:
            # f.write(f"for index: {index} at iteration {i} the changes array is {temp}\n")
            f.write(f"for index: {index} density  list is {average_density}\n")
    # graph_time(time)

#testing for 1000 iters on jojo => recall, alpha =1 (less hook)
#for index: 13 density  list is 
l = [8.350531192150143, 8.836733070870311, 8.15026908322514, 8.169314109471728, 8.120657781738904, 7.8387408047366165, 7.871288398636948, 7.704663253001619, 7.7222812864561, 7.638716922882653, 7.529145282420763, 7.4196367780568195, 7.293277663501195, 7.294934048231607]

#testing Betz => recall, alpha =0 10,000 for 38 iters
# for index: 38 density  list is 
l1 = [2.894564798436663, 2.742658473504928, 2.9500032091398998, 2.9716900719955452, 2.9626442698783673, 2.9574257539860693, 2.9375175098130635, 2.920897184747449, 2.9207207787799887, 2.906564512562613, 2.9060831001510286, 2.898443841835443, 2.892349929487068, 2.8898419706793708, 2.877596519273487, 2.872609737382494, 2.8692976791230196, 2.863144291075841, 2.847519525391978, 2.8347488911874694, 2.8278018865221695, 2.818073908160802, 2.799248385758665, 2.798484213955967, 2.784911057895057, 2.7677313715197864, 2.759365109638811, 2.7410102599541686, 2.734705311159419, 2.7116538031249826, 2.6936397658142583, 2.667094037196456, 2.6370377826340183, 2.6112972957597895, 2.588518300310813, 2.5570871450771566, 2.508867230824737, 2.467657775280267, 2.415364841269837]
