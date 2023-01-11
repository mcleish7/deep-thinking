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
import seaborn as sns

def get_net(device, type="top"):
    """
    Returns the 'top' or 'bottom' network reffering to their positions in recovery time from a peturbation, in eval mode

    Args:
        device (str): the device we are working on
        type (str, optional): what network you want. Two options top or bottom, meaning the top or bottom line of the two. Defaults to "top".

    Returns:
        torch.nn: the neural network
    """
    if type == "top":
        # Betz => recall, alpha =0
        file = "batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida/model_best.pth"
    else: #alpha=0.8
        file = "mismatch/outputs/prefix_sums_ablation/training-gowaned-Ayla/model_best.pth"

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

def net_out_to_bits(output,target, log = False, graph = False): #output from the net and the target bit string
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
            j = self.j #between 0 and 48
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -20.0 #means that 0 will not be returned as it is less than the 1 index, i.e. a bitflip
                else:
                    output[0,1,j] = 20.0

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
    bestind = np.argmax(correct[50:]) #only looks for maximum after peturbation
    return bestind 

def graph_time(arr1,arr2):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr1 (list): list of the times to recover of the Recall net being used
        arr2 (list): list of the times to recover of the Recall Progressive net being used
    """
    plt.clf()
    plt.plot(arr1, linewidth = '3.0', label = "Recall")
    plt.plot(arr2, linewidth = '3.0', label = "Recall Prog")
    plt.title('Iterations to recover from a single bit perturbation')
    plt.xlabel("Index to be flipped")
    plt.ylabel("Number of iterations to recover")
    plt.legend(loc="upper right")
    plt.yticks([0,26,5,10,25,15,20])
    save_path = os.path.join("test_time","test_time_correctness_2.png")
    plt.savefig(save_path)

def num_diff(arr1,arr2):
    """
    Finds the number of differences in elements of the same index between the two inputs
    Note: assumes two arrays are of same length
    Args:
        arr1 (list): first list to compare
        arr2 (list): second list to compare

    Returns:
        int: the number of differences between the inputs
    """
    count = 0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1
    return count

def convert_to_bits(input): #in shape is [1, 300, 2, 48]
    """Convert the input string to a bits stored in an array

    Args:
        input (tensor): the  array to convert

    Returns:
        golden_label (numpy array): the input string to a bits stored in an array
    """
    # print("in argmax is ",input[0,0].argmax(0))
    predicted = input.clone().argmax(2)
    predicted = predicted.reshape([1,300,48])
    # print("predicted=",predicted[0,0])
    # print("same ? ", torch.equal(input[0,0].argmax(0),predicted[0,0]))
    # golden_label = predicted.view(predicted.size(0), -1)
    return predicted

def main():
    """
    Runs the peturbation with the input commmand line peratmeter for which net is selected
    """
    # parser = argparse.ArgumentParser(description="Time parser")
    # parser.add_argument("--which_net", type=str, default="prog", help="choose between prog or non-prog, defaults to prog")
    # args = parser.parse_args()

    os.chdir("/dcs/large/u2004277/deep-thinking/") # changing back to the top directory as this method can be called from bash scripts in other directories
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parameters for pytorchfi model
    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]
    print("now going into loop")
    inputs,targets = get_data(device)
    with torch.no_grad(): # we are evaluating so no grad needed
        time = [] # store for the averaged values
        for index in range(0,41): # index of which bit is to be changed
            average = []
            top_net = get_net(device, type = "top")
            pfi_model_top = custom_func(index,top_net, 
                                batch_size,
                                input_shape=[channels,width],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
            bottom_net = get_net(device, type = "bottom")
            pfi_model_bottom = custom_func(index,bottom_net, 
                                batch_size,
                                input_shape=[channels,width],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
            for i in range(0,inputs.size(0)):
                input = inputs[i].unsqueeze(0) # have to unsqueeze to simulate batches
                target = targets[i].unsqueeze(0) 
                inj_top = pfi_model_top.declare_neuron_fi(function=pfi_model_top.flip_all) 
                inj_bottom = pfi_model_bottom.declare_neuron_fi(function=pfi_model_bottom.flip_all) 
                inj_output_top = inj_top(input)
                # print("outputs shape is ",inj_output_top.shape)
                inj_output_bottom = inj_bottom(input)
                bits_top = convert_to_bits(inj_output_top).squeeze().cpu().detach().numpy()
                bits_bottom = convert_to_bits(inj_output_bottom).squeeze().cpu().detach().numpy()
                difference = np.not_equal(bits_bottom,bits_top).sum(axis=1)
                # print("for index ", i, "difference is ",difference[50:])
                average.append(difference[50:150]) #take 100 indecies after peturb
            average = np.array(average)
            mean = np.mean(average,axis=0)
            time.append(mean)
            np_time = np.array(time)
            # print(np_time.shape)
            # print(np_time)
            np.save("compare_time_3.npy",np_time) #save data for later analysis

# if __name__ == "__main__":
#     main()


def heat_map_plot():
    """
    Loads the data generated by the main method and creates a heatmap to show where the differences are between the two networks
    """
    mat1 = np.load("compare_time.npy")
    mat2 = np.load("compare_time_2.npy")
    mat = np.concatenate((mat1,mat2))
    mat = np.transpose(mat)[:35]
    ax = sns.heatmap(mat, linewidth=0.5)
    ax.invert_yaxis()
    ax.set(xlabel='Index of bit flipped', ylabel='Iteration after peturbation', title="Average number of bits different")
    plt.savefig("compare_time_heatmap", bbox_inches="tight")
heat_map_plot()