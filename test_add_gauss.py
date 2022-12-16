import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import sys
import os 
from deepthinking import models as models
from deepthinking import utils as dt
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse


def get_net(device, which_net="prog"):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        which_net (str, optional): Set to prog if want the progressive recall network. Defaults to "prog".
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if which_net == "prog": 
        name = "enraged-Jojo" # Jojo => recall, alpha =1 
    else:
        name = "peeling-Betzaida" # Betz => recall, alpha =0
    file = f"batch_shells_sums/outputs/prefix_sums_ablation/training-{name}/model_best.pth"

    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30)
    state_dict = torch.load(file, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_data(device):
    """
    Gets bit strings of length 48 from the local file and augments them to be the same how the DataLoader would input them
    Args:
        device (str): the device to store the output tensors on
    Returns:
        input, target (tensor,tensor): the input and taget datasets as tensors on the device passed in
    """
    data = torch.load("batch_shells_sums/data/prefix_sums_data/48_data.pth").unsqueeze(1) - 0.5
    target = torch.load("batch_shells_sums/data/prefix_sums_data/48_targets.pth")
    input = data.to(device, dtype=torch.float) #to account for batching in real net
    target = target.to(device, dtype=torch.float)
    return input, target

def convert_to_bits(input): 
    """Convert the input string to a bits stored in an array

    Args:
        input (tensor): the numpy array to convert

    Returns:
        golden_label (numpy array): the input string to a bits stored in an array
    """
    predicted = input.clone().argmax(1)
    golden_label = predicted.view(predicted.size(0), -1)
    return golden_label

def graph_progress(arr):
    """
    Graph the input array as a line graph
    This method is only called in the net_out_to_bits method so the title and save path are fixed
    Args:
        arr (numpy.array or list): the array/list to be graphed
    """
    plt.plot(arr)
    plt.title('Values of correct array')
    save_path = os.path.join("test_time","test_gauss_graph.png")
    plt.savefig(save_path)

def net_out_to_bits(output,target, log = False, graph = False): 
    """
    For a testing run of a NN finds the output of the NN with the best accuracy and what this output was.
    Can also store the run in a json file and plot it.

    Args:
        output (tensor): output from a run of the NN
        target (tensor): the target of the NN run
        log (bool, optional): If set to True, saves the number of correct bits at iteration i for each iteration of the NN in a json file. Defaults to False.
        graph (bool, optional): If set to True, graphs the number of correct bits per iteration of the NN using the graph_progress method. Defaults to False.

    Returns:
        numpy.array, int: numpy array of the closest prediction of the NN to the target and the iteration that this was at. Takes the first index if reaches 100% accuracy
    """
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
    

class custom_func(fault_injection):
    """
    Custom peturbation class to peturb the input of the prefix sums NNs at the 50 iteration
    Inherits:
        fault_injection (class): pytorchfi.core.fault_injection

    Attributes
    ----------
    j : int
        the bit to be peturbed in the string

    Methods
    -------
    flip_all(self, module, input, output)
        called at each iteration of the NN to add gaussian noise
    """
    def __init__(self,model, batch_size, **kwargs):
        """constructor for custom_func

        Args:
            in_j (int): the bit to be peturbed
            model (_type_): the network
            batch_size (int): batch size for the network
        """
        super().__init__(model, batch_size, **kwargs)

    def flip_all(self, module, input, output):
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer

        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tensor): the output of the layer
        """
        if (self.get_current_layer() < 24) and (self.get_current_layer() >= 8):
            gauss = torch.randn(output.size()).to(device)
            output = output+10*(gauss)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def count_to_correct(output,target): 
    """
    finds the number of iterstions to recover from peturbation
    Note: not used in this file, but kept for future user interest

    Args:
        output (tensor): _description_
        target (tensor): _description_

    Returns:
       int: _description_
    """
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    return bestind 

def get_corrects_array(output,target):
    """
    outputs an array of the number of indicies correct at each iteration
    
    Args:
        output (tensor): the output of the run of the net
        target (tensor): the target of the run of the net

    Returns:
        numpy.array: the number of indicies correct at each iteration
    """
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
    correct = corrects.cpu().detach().numpy()
    return correct

def graph_time(arr1,arr2):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays

    Args:
        arr1 (list/np.array): list of the times to recover of the Recall net being used
        arr2 (list/np.array): list of the times to recover of the Recall Progressive net being used
    """
    plt.clf()
    arr1 = np.asarray(arr1)*(100/48) # to make percentages
    arr2 = np.asarray(arr2)*(100/48)
    plt.plot(arr2, linewidth = '1.0', label = "Recall")
    plt.plot(arr1, linewidth = '1.0', label = "Recall Prog")
    plt.title('Number of iterations to solution \n with Gaussian noise added at iterations 1 and 2')
    plt.xlabel("Test-Time iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.ylim([0, 101])
    save_path = os.path.join("test_time","test_add_gauss.png")
    plt.savefig(save_path)

os.chdir("/dcs/large/u2004277/deep-thinking/")
cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 500
channels = 1
width = 400
layer_types_input = [torch.nn.Conv1d]

net = get_net(device, which_net="prog")
inputs,targets = get_data(device)
with torch.no_grad():
    corrects = []
    for i in range(0,inputs.size(0)):
        input = inputs[i].unsqueeze(0)
        target = targets[i].unsqueeze(0)
        output = net(input)
        pfi_model = custom_func(net, 
                                batch_size,
                                input_shape=[channels,width],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )

        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)
        inj_output = inj(input)
        corrects.append(get_corrects_array(inj_output,target))

corrects_1 = np.array(corrects)
mean_corrects_1 = np.mean(corrects_1, axis=0) # take the mean of the values

# re-run testing for non-progressive net
net = get_net(device, which_net="non-prog")
inputs,targets = get_data(device)
with torch.no_grad():
    corrects = []
    for i in range(0,inputs.size(0)):
        input = inputs[i].unsqueeze(0)
        target = targets[i].unsqueeze(0)
        output = net(input)
        pfi_model = custom_func(net, 
                                batch_size,
                                input_shape=[channels,width],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )

        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)
        inj_output = inj(input)
        corrects.append(get_corrects_array(inj_output,target))

corrects_2 = np.array(corrects)
mean_corrects_2 = np.mean(corrects_2, axis=0)

mean_corrects_1 = np.insert(mean_corrects_1, 0, 0) # adds 0 to the start
mean_corrects_2 = np.insert(mean_corrects_2, 0, 0)
print(mean_corrects_1)
print(mean_corrects_2)
graph_time(mean_corrects_1,mean_corrects_2)


mean_corrects_1 = [ 0., 25.0679, 26.6453, 28.1859, 29.6425, 31.1056, 32.5796, 34.0042, 35.2958,\
 36.571,  37.7723, 38.9306, 40.0978, 41.2588, 42.434,  43.563,  44.576,  45.4299,\
 46.2983, 47.0347, 47.3775, 47.7029, 47.8993, 47.9455, 47.9824, 47.9913, 47.9954,\
 47.9984, 47.9985, 47.9993, 47.9996, 47.9998, 47.9999, 47.9999, 48., 47.9999,\
 48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48.]

mean_corrects_2 = [0., 24.3598, 24.8983, 25.5996, 26.4618, 27.3484, 28.2358, 29.157,  30.0687,\
 30.9608, 31.8518, 32.729, 33.6329, 34.5249, 35.4035, 36.3208, 37.2139, 38.1044,\
 38.9735, 39.8351, 40.6863, 41.5078, 42.362,  43.2369, 44.0855, 44.9111, 45.678,\
 46.3546, 46.9115, 47.3548, 47.6744, 47.8672, 47.9606, 47.9923, 47.9992, 47.9998,\
 48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48., 48., 48., 48., 48., 48.,48., 48., 48., 48.]
# both length 301