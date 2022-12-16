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

def graph_time(arr0,arr01,arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr(n) (list): list of the times to recover of the alpha=n/10 net being used
    """
    plt.clf()
    plt.plot(arr0, linewidth = '1.0', label = "0.0")
    plt.plot(arr01, linewidth = '1.0', label = "0.01")
    plt.plot(arr1, linewidth = '1.0', label = "0.1")
    plt.plot(arr2, linewidth = '1.0', label = "0.2")
    plt.plot(arr3, linewidth = '1.0', label = "0.3")
    plt.plot(arr4, linewidth = '1.0', label = "0.4")
    plt.plot(arr5, linewidth = '1.0', label = "0.5")
    plt.plot(arr6, linewidth = '1.0', label = "0.6")
    plt.plot(arr7, linewidth = '1.0', label = "0.7")
    plt.plot(arr8, linewidth = '1.0', label = "0.8")
    plt.plot(arr9, "--", linewidth = '1.0', label = "0.9")
    plt.plot(arr10, "--", linewidth = '1.0', label = "1.0")
    plt.title('Iterations to recover')
    plt.legend(loc="upper right",prop={'size': 8})
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlim([-0.5,45])
    # plt.yticks([0,26,5,10,25,15,20])
    plt.yticks([3,5,10,15,20])
    save_path = os.path.join("test_time","test_mismatch_time_correctness_2.png")
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

#for index 0.01:
l011 = [10.3156, 12.3392, 15.1635, 16.0805, 15.4838, 14.9087, 14.7462, 14.4749, 14.2972, 13.8323, 13.73, 13.3576, 13.0225, 12.8171, 12.4871, 12.514, 11.9222, 11.7036, 11.4577, 11.2581, 10.8513, 10.5424, 10.4047, 10.019, 9.8045, 9.5763, 9.2741, 8.9566, 8.6958, 8.3057, 8.11, 7.7929, 7.531, 7.2778, 6.9837]
l012 = [6.6743, 6.4322, 6.1498, 5.8914, 5.5296]
l01 = l011 + l012

# for index 0.1:
l11 = [15.2739, 12.0973, 14.6035, 11.8572, 12.8455, 12.3735, 12.1936, 11.9801, 11.7079, 11.4299, 11.2314, 10.8221, 10.6674, 10.3556, 10.1579, 9.959, 9.7739, 9.4513, 9.1935, 8.8998, 8.7371, 8.44, 8.2186, 8.0045, 7.6584, 7.4866, 7.2773, 7.0027, 6.7449, 6.5592, 6.1613, 5.9436, 5.671, 5.3641, 5.1801]
l12 = [4.9681, 4.6728, 4.4116, 4.1683, 3.9348]
l1 = l11 +l12

#for index 0.2:
l21 = [11.5842, 15.3502, 17.6201, 16.5837, 17.5766, 16.9705, 16.8695, 16.4734, 16.0436, 15.7038, 15.5311, 15.2334, 14.8227, 14.567, 14.1204, 14.013, 13.5292, 13.1629, 12.7815, 12.5513, 12.2482, 11.8385, 11.6202, 11.243, 10.9988, 10.5085, 10.3192, 9.9325, 9.5159, 9.2203, 8.8349, 8.483, 8.1235, 7.7819, 7.4308]
l22 = [7.0491, 6.6779, 6.3296, 5.9807, 5.5463]
l2 = l21+l22

#for index 0.3:
l31 = [15.3406, 14.8666, 14.8177, 15.6047, 15.6742, 15.7452, 15.4569, 14.8674, 14.6863, 14.1978, 14.2221, 13.9014, 13.52, 13.3189, 12.8938, 12.6709, 12.174, 12.2708, 11.8266, 11.5006, 11.1833, 10.9521, 10.7211, 10.3281, 10.1386, 9.6848, 9.5377, 9.2434, 8.9042, 8.5797, 8.3227, 7.9888, 7.6193, 7.3458, 6.987]
l32 = [6.694, 6.4279, 6.0516, 5.7919, 5.4095]
l3 = l31+l32

#for index 0.4:
l41 = [12.1917, 12.5917, 13.3297, 12.393, 12.7258, 12.5114, 12.4629, 12.0591, 11.886, 11.631, 11.5766, 11.2877, 11.0672, 10.9445, 10.5777, 10.4138, 10.2563, 10.1335, 9.7631, 9.4921, 9.3046, 9.0427, 8.8761, 8.6486, 8.4816, 8.1028, 7.9876, 7.7722, 7.4667, 7.2861, 7.0445, 6.7492, 6.5239, 6.2519]
l42 = [6.0443, 5.7809, 5.6139, 5.2923, 5.0796, 4.786]
l4 = l41+l42

# prog
# for index: 36 the time array is 
prog1 = [16.1856, 15.5111, 16.3486, 15.4524, 15.5068, 15.0618, 14.9197, 14.4535, 14.2271, 13.9523, 13.5131, 13.2493, 12.7945, 12.5634, 12.1517, 11.9928, 11.3922, 11.1332, 10.9404, 10.5358, 10.1458, 9.7926, 9.6382, 9.1171, 8.831, 8.5497, 8.1151, 7.9084, 7.5659, 7.1674, 6.9791, 6.665, 6.2769, 5.9907, 5.7131, 5.4087, 5.1498]
# for index: 39 the time array is 
prog2 = [5.1498, 4.9016, 4.5934, 4.2871]
prog = prog1+prog2

#non prog
# for index: 36 the time array is 
nprog1 =[13.1338, 14.0909, 16.5159, 18.9491, 18.6721, 18.2342, 17.807, 17.2267, 16.8972, 16.5704, 16.2457, 16.0122, 15.6696, 15.3622, 15.1253, 14.7834, 14.5677, 14.2961, 13.9309, 13.6141, 13.2166, 13.0427, 12.7052, 12.3991, 12.0407, 11.7189, 11.4802, 11.0965, 10.7525, 10.444, 10.0743, 9.6769, 9.2973, 8.9339, 8.6153, 8.2189, 7.8931]
# for index: 39 the time array is 
nprog2 = [7.8931, 7.4263, 7.0412, 6.6338]
nprog = nprog1+nprog2

# graph_time(l0,l01,l1,l2,l3,l4,l5,l6,l7,l8,l9,l10)
graph_time(nprog,l01,l1,l2,l3,l4,l5,l6,l7,l8,l9,prog)