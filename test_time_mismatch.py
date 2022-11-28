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

def get_net(which_net,device):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        which_net (int): The alpha value of the network times 10, e.g. which_net=5 maps to a net with alpha value 0.5
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if which_net == 5:
        name = "freckly-Lonnell"
    elif which_net == 6:
        name = "stunning-Hank"
    elif which_net == 7:
        name = "faddy-Pual"
    elif which_net == 8:
        name = "gowaned-Ayla"
    else: #i.e. is 0.9
        name = "crudest-Tanda"
    full_path = f"mismatch/outputs/prefix_sums_ablation/training-{name}/model_best.pth"
    state_dict = torch.load(full_path, map_location=device)
    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300)
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
    count = 0
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

    # define your own function
    def flip_all(self, module, input, output): 
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer

        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tuple): the output of the layer
        """
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

def main():
    """
    Runs the peturbation with the input commmand line peratmeter for which net is selected
    """
    parser = argparse.ArgumentParser(description="Time parser")
    parser.add_argument("--which_net", type=int, default=5, help="choose the alpha of the net required")
    args = parser.parse_args()

    os.chdir("/dcs/large/u2004277/deep-thinking/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = get_net(args.which_net,device)
    # parameters for pytorchfi model
    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]

    print("now going into loop")
    inputs,targets = get_data(device)
    with torch.no_grad():
        time = []
        for index in range(10,40):
            average = []
            for i in range(0,inputs.size(0)):
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
            file_name = f"time_peturb_list_{args.which_net}.txt"
            file_path = os.path.join("test_time",file_name)
            with open(file_path, 'r+') as f:
                f.write(f"for index: {index} the time array is {time}")

# if __name__ == "__main__":
#     main()

def graph_time(arr0,arr1,arr2,arr3,arr4,arr5,arr10):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr(n) (list): list of the times to recover of the alpha=n/10 net being used
    """
    plt.clf()
    plt.plot(arr0, linewidth = '3.0', label = "0.0")
    plt.plot(arr1, linewidth = '3.0', label = "0.5")
    plt.plot(arr2, linewidth = '3.0', label = "0.6")
    plt.plot(arr3, linewidth = '3.0', label = "0.7")
    plt.plot(arr4, linewidth = '3.0', label = "0.8")
    plt.plot(arr5, linewidth = '3.0', label = "0.9")
    plt.plot(arr10, linewidth = '3.0', label = "1.0")
    plt.title('Iterations to recover')
    plt.legend(loc="upper right")
    plt.yticks([0,26,5,10,25,15,20])
    save_path = os.path.join("test_time","test_mismatch_time_correctness.png")
    plt.savefig(save_path)

# All of the output data is stored in text files, I have moved it to here to graph and so it can be seen in its raw format
# Runs oftern take more than two days, hence the split in the lists

# for 0.5:
l51 = [14.8176, 13.2959, 13.9689, 14.8747, 14.853, 14.3676, 14.1875, 13.8164, 13.5384, 13.3119]
l52 = [13.192, 12.8912, 12.6419, 12.455, 12.0878, 12.0119, 11.4937, 11.4611, 11.1552, 10.7682, 10.5393, 10.2569, 10.094, 9.8167, 9.6313, 9.34, 9.0786, 8.7576, 8.5151, 8.2099, 8.0048, 7.7308, 7.4295, 7.1984, 6.9771, 6.6661, 6.4059, 6.1599, 5.8936, 5.6001]
l5 = l51+l52

# for 0.6:
l61 = [9.3085, 10.3761, 11.2802, 10.4377, 10.5848, 10.4299, 10.1049, 9.9819, 9.7614, 9.6285]
l62 = [9.5093, 9.2107, 9.1643, 8.8823, 8.8231, 8.5303, 8.3725, 8.2285, 8.0799, 7.8115, 7.6546, 7.4623, 7.279, 7.1319, 6.9703, 6.7702, 6.5828, 6.3335, 6.1747, 5.9514, 5.7582, 5.59, 5.3635, 5.186, 5.0293, 4.8519, 4.6223, 4.4445, 4.2572, 4.0763]
l6 = l61+l62

# for 0.7:
l71 = [7.4427, 8.7147, 8.7455, 8.1812, 8.4095, 8.2813, 8.0313, 7.8941, 7.7916, 7.7862]
l72 = [7.5654, 7.5478, 7.4356, 7.3429, 7.1804, 7.0958, 7.1258, 6.9335, 6.8393, 6.7082, 6.6213, 6.4753, 6.3806, 6.1937, 6.0148, 5.9719, 5.8272, 5.6555, 5.5412, 5.4545, 5.2382, 5.0934, 4.9541, 4.7572, 4.5841, 4.4994, 4.2927, 4.0343, 3.8877, 3.7184]
l7 = l71+l72

# for 0.5:
l81 = [8.42, 5.1115, 6.914, 6.2542, 6.2126, 6.542, 6.3429, 6.358, 6.2839, 6.2597]
l82 = [6.0986, 6.1008, 6.0576, 5.9729, 5.9198, 5.8701, 5.8492, 5.5524, 5.6518, 5.5899, 5.5575, 5.3769, 5.3281, 5.2861, 5.1361, 5.126, 5.0006, 4.8715, 4.8574, 4.6982, 4.5662, 4.5383, 4.4131, 4.3134, 4.2108, 4.0727, 3.9853, 3.8711, 3.7905, 3.6211]
l8 = l81+l82

# for 0.9:
l91 = [8.7483, 10.2486, 10.1857, 10.2617, 10.2634, 10.209, 10.0174, 9.8483, 9.7013, 9.493]
l92 = [9.4598, 9.1231, 9.0177, 8.8518, 8.7085, 8.4107, 8.2773, 8.1505, 7.998, 7.8195, 7.6068, 7.4807, 7.2258, 7.1361, 6.8788, 6.6843, 6.5464, 6.3344, 6.2028, 5.9924, 5.8071, 5.629, 5.4315, 5.2355, 5.0184, 4.8807, 4.6546, 4.4152, 4.2949, 4.0279]
l9 = l91+l92



# Betz data => recall, alpha =0
l01 = [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114] 
l02 = [9.9924, 9.4904, 8.9798]
l0 = l01+l02

# Jojo data => recall, alpha =1
l101 = [6.3722, 6.0566, 5.6513]
l102 = [17.8872, 19.4591, 19.941, 19.912, 19.8543, 19.2299, 18.8148, 18.4023, 18.0458, 17.6382, 17.1686, 16.7601, 16.27, 15.8989, 15.4843, 15.1598, 14.6714, 14.3212, 13.8789, 13.4322, 13.014, 12.6209, 12.1961, 11.7285, 11.3529, 10.9505, 10.512, 10.1154, 9.707, 9.2398, 8.893, 8.5136, 8.1195, 7.723, 7.3973, 7.024, 6.6765]
l10 = l102+l101

graph_time(l0,l5,l6,l7,l8,l9,l10)