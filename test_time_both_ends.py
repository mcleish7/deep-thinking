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

def get_net(device, type="prog"):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        type (str, optional): Set to prog if want the progressive recall network. Defaults to "prog".
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if type == "prog": 
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
    input = data.to(device, dtype=torch.float) 
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
    save_path = os.path.join("test_noise_outputs","test_noise_correctness.png")
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
        called at each iteration of the NN to flip the specified bit
    """
    j = 0 
    def __init__(self, in_j,model, batch_size, **kwargs):
        """constructor for custom_func

        Args:
            in_j (int): the bit to be peturbed
            model (_type_): the network
            batch_size (int): batch size for the network
        """
        super().__init__(model, batch_size, **kwargs)
        self.j = in_j

    def flip_all(self, module, input, output): 
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer

        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tensor): the output of the layer
        """
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400):
            j = self.j #between 0 and 23 as flipping both ends
            otherj = 47-j # index for the second bit to be flipped
            # print("output size is ",output.size())
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -20.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,j] = 20.0
                if output[0,i,otherj] > 0.0:
                    output[0,0,otherj] = -20.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,otherj] = 20.0
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def count_to_correct(output,target): 
    """
    Counts the number of iterations until the network finds the correct output after peturbation

    Args:
        output (tensor): the output of the NN run
        target (tensor): the target of the run

    Returns:
        int: the number of iterations it took to recover from peturbation
    """
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
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr1 (list): list of the times to recover of the Recall net being used
        arr2 (list): list of the times to recover of the Recall Progressive net being used
    """
    plt.clf()
    plt.plot(arr1, linewidth = '2.0', label = "Recall")
    plt.plot(arr2, linewidth = '2.0', label = "Recall Prog")
    plt.title('Iterations to recover from a multi bit perturbation')
    plt.xlabel("Index to be flipped, both ith and (47-i)th index flipped")
    plt.ylabel("Number of iterations to recover")
    plt.legend(loc="upper right")
    plt.yticks([0,5,10,25,15,20])
    save_path = os.path.join("test_time","test_time_both_ends.png")
    plt.savefig(save_path)

def main():
    """
    Runs the peturbation with the input commmand line peratmeter for which net is selected
    """
    parser = argparse.ArgumentParser(description="Time parser")
    parser.add_argument("--which_net", type=str, default="prog", help="choose between prog or non-prog, defaults to prog")
    args = parser.parse_args()
    os.chdir("/dcs/large/u2004277/deep-thinking/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]

    net = get_net(device, args.which_net)
    inputs,targets = get_data(device)

    with torch.no_grad():
        time = []
        for index in range(0,24): #difference to test_time as now only going to half the bit string 
            average = [] # store for the averaged values
            for i in range(0,inputs.size(0)):
                input = inputs[i].unsqueeze(0) # to simulate batches
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
            file_path = os.path.join("test_time","time_both_ends_tracker.txt")
            with open(file_path, 'r+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
                f.write(f"for index: {index} the time array is {time}")

# if __name__ == "__main__":
#     main()

# All of the output data is stored in text files, I have moved it to here to graph and so it can be seen in its raw format
# Runs oftern take more than two days, hence the split in the lists

# for Jojo => recall, alpha =1
# for index: 23 the time array is 
j = [20.2819, 20.0735, 18.9922, 18.8547, 18.6998, 18.2775, 18.0904, 17.6643, 17.2837, 16.9909, 16.5707, 16.2337, 15.7949, 15.4522, 15.0258, 14.7135, 14.2244, 13.8056, 13.3625, 12.7967, 12.2584, 11.6712, 11.0679, 9.686]

# for Betz => recall, alpha =0
# for index: 23 the time array is 
b = [15.8195, 16.4843, 19.5093, 21.2553, 21.0011, 20.7749, 20.3372, 19.9753, 19.7307, 19.449, 19.3567, 19.1141, 18.8925, 18.6742, 18.4172, 18.1089, 17.81, 17.5339, 17.6947, 17.8454, 17.4929, 17.1599, 16.6152, 15.6152]

graph_time(b,j)