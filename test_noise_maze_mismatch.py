import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import deepthinking.models as models
import deepthinking.utils as dt
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from easy_to_hard_plot import plot_maze
from easy_to_hard_plot import MazeDataset
import argparse

def get_net(device, which_net):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        which_net (int): The alpha value of the network times 10, e.g. which_net=5 maps to a net with alpha value 0.5
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if which_net == 1:
        name = "inmost-Quenten"
    elif which_net == 2:
        name = "yolky-Dewaun"
    elif which_net == 3:
        name = "tented-Arlena"
    elif which_net == 4:
        name = "cormous-Andreah"
    elif which_net == 5:
        name = "stalkless-Terricka"
    elif which_net == 6:
        name = "exchanged-Nyasia"
    elif which_net == 7:
        name = "feeblish-Ernesto"
    elif which_net == 8:
        name = "cosher-Taneika"
    elif which_net == 9:
        name = "praising-Kimberely"
    else:
        name = "heating-Mihcael"
    full_path = f"mismatch/outputs/mazes_ablation/training-{name}/model_best.pth"
    state_dict = torch.load(full_path, map_location=device)
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net


def get_data(n=10, size=13):
    #n is the number of elemtns wanted
    # returns an [n,3,32,32] shape tensor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")
    target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")
    if size == 59:
        data = np.load("batch_reproduce_2.2/data/maze_data_test_59/inputs.npy")
        target = np.load("batch_reproduce_2.2/data/maze_data_test_59/solutions.npy")
    # print("data shape is ",data.shape)
    a = data[:n]
    a = torch.from_numpy(a)
    # print("a before shape",a.shape)
    input = a.to(device, dtype=torch.float)#.unsqueeze(0) #to account for batching in real net
    # print("a after shape",input.shape)
    b = target[:n]
    t = torch.from_numpy(b)
    t = t.to(device, dtype=torch.float)
    target = t#.unsqueeze(0)
    return input, target, a

def convert_to_bits(output, input):
    """
    Converts the output of the net to its prediciton

    Args:
        output (tensor): the output of the net
        input (tensor): the input to the net

    Returns:
        tensor: the prediction of the net
    """
    output = output.unsqueeze(0).to(device)
    # print("ouput shape is ",output.shape)
    predicted = output.clone().argmax(1)
    # print("predicted 2 shape is ",predicted.shape)
    predicted = predicted.view(predicted.size(0), -1)
    # print("predicted 3 shape is ",predicted.shape)
    # input = input.squeeze()
    # print("input shape is ",input.max(1)[0].view(input.size(0), -1).shape)
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1)) #used to map the output into only paths that exist in the maze
    # print("golden label shape is", golden_label.shape)
    return golden_label

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
ex = torch.zeros((3, 1, 400), dtype=torch.float)
input, target, a = get_data()
iters =500
corrects = torch.zeros(iters)

class custom_func(fault_injection):

    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        layer_from = 50 #for small GPU's use 25 or less, for larger ones we can use the full result of 50
        layer_to = 51
        if (self.get_current_layer() >= (layer_from*7)) and (self.get_current_layer() <= ((layer_to*7)+1)): # a nice observation here is the direct relation ot the size of the recurrent module
            output[:] = torch.zeros(output.shape) # puts all outputs from the layer to 0
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

def graph_helper(output,input,target):
    """
    Very much like convert to bits but specifically trimmed down and molded to service the graph_two method

    Args:
        output (Torch.tensor): the output from a run of a net
        input (Torch.tensor): the input to the net
        target (Torch.tensor):the target of the net

    Returns:
        numpy.array: the number of bits which were predicted correctly at each iteration of the net
    """
    output = output.clone().squeeze()
    corrects = torch.zeros(output.shape[0])
    for i in range(output.shape[0]): # goes through each iteration
        # print("in g haleper output 1 shape is ",output.shape)
        outputi = output[i]
        # print("in g haleper output 2 shape is ",outputi.shape)
        # print("output i shape is ",outputi.shape)
        # outputi = torch.from_numpy(outputi)
        # print("output i shape is ",outputi.shape)
        golden_label = convert_to_bits(outputi, input)
        target = target.view(target.size(0), -1)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() # counts the number that are the same
    correct = corrects.cpu().detach().numpy()
    return correct

def graph_maze_mismatch(runs, input, target, n, size):
    plt.clf()
    alphas = ["0.01","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
    denom = 1024.0
    if size != 13:
        denom = 15376.0
    # print("len runs is ",len(runs))
    # print(type(runs))
    for i in range(0,len(runs)):
        run = runs[i]
        # print("run is type",type(run))
        # print(run.shape)
        plt.plot(run*(100.0/denom), linewidth = '1.0', label = alphas[i])
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    file_name = f"maze{size}_mismatch_{n}.png"
    # file_name = "test.png"
    save_path = os.path.join("test_noise_outputs",file_name)
    plt.savefig(save_path, dpi=500)

def main_module(number=100, input_size = 13):
    # PyTorchFi parameters for the maze nets
    batch_size = 1
    channels = 3
    width = 128
    height = width
    layer_types_input = [torch.nn.Conv2d]

    nets_list = [-1,1,2,3,4,5,6,7,8,9]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    averages = []
    inputs, targets, a = get_data(n=number, size = input_size)
    for alpha in nets_list:
        print(f"on alpha: {alpha}")
        outputs = []
        with torch.no_grad():
            net = get_net(device, alpha)
            pfi_model = custom_func(net, 
                                    batch_size,
                                    input_shape=[channels,width,height],
                                    layer_types=layer_types_input,
                                    use_cuda=True
                                )
            for i in range(0,inputs.size(0)):
            # for i in range(0,2):
                # print("inputs shape",inputs.shape)
                # print("inputs i shape",inputs[i].shape)
                input = inputs[i].unsqueeze(0) #have to unsqueeze to simulate batches
                target = targets[i].unsqueeze(0) 
                inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)
                out = inj(input)
                converted = graph_helper(out,input,target)
                # print("converted shape is ",converted.shape)
                # print(converted[10])
                # print(converted[1])
                # print("out type is ",type(out))
                # print(type(converted))
                outputs.append(converted)
                # outputs.append(out.squeeze().cpu().detach().numpy())
            # print("outputs type is ",type(outputs))
            outputs = np.array(outputs)
            # print("before",outputs.shape)
            average = np.mean(outputs,axis = 0)
            averages.append(average)
            # print("after",average.shape)
            # np.save("test_noise_outputs/test_maze_mismatch_averages.npy",np.array(averages))
    graph_maze_mismatch(averages, input, target, number, input_size)

def main():
    # ns = [10,50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
    # ns = [10,50,100,200,300,400,500]
    ns = [5000,10000]
    maze_dim = 59
    for n in ns:
        main_module(number = n, input_size=maze_dim)

# def saved():
#     averages = np.load("test_noise_outputs/test_maze_mismatch_averages.npy")
#     print("averages shape is ",averages.shape)
#     number = 10000
#     input_size = 59
#     graph_maze_mismatch(averages, input, target, number, input_size)
if __name__ == "__main__":
    main()


# def sums_main():
#     """
#     Runs the peturbation with the input commmand line peratmeter for which net is selected
#     """
#     parser = argparse.ArgumentParser(description="Time parser")
#     parser.add_argument("--which_net", type=str, default="prog", help="choose between prog or non-prog, defaults to prog")
#     args = parser.parse_args()

#     os.chdir("/dcs/large/u2004277/deep-thinking/") # changing back to the top directory as this method can be called from bash scripts in other directories
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # parameters for pytorchfi model
#     batch_size = 500
#     channels = 1
#     width = 400
#     layer_types_input = [torch.nn.Conv1d]

#     net = get_net(device, type = args.which_net)
#     print("now going into loop")
#     inputs,targets = get_data(device)
#     with torch.no_grad(): # we are evaluating so no grad needed
#         time = [] # store for the averaged values
#         for index in range(0,40): # index of which bit is to be changed
#             average = []
#             for i in range(0,inputs.size(0)):
#                 input = inputs[i].unsqueeze(0) # have to unsqueeze to simulate batches
#                 target = targets[i].unsqueeze(0) 
#                 pfi_model_2 = custom_func(index,net, 
#                                         batch_size,
#                                         input_shape=[channels,width],
#                                         layer_types=layer_types_input,
#                                         use_cuda=True
#                                     )

#                 inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all) # run the model, the number of iterations is controlled in by the default value in the forward call of each model
#                 inj_output = inj(input)
#                 average.append(count_to_correct(inj_output,target))
#             mean = sum(average) / len(average)
#             time.append(mean)
#             name = f"time_list_tracker_{args.which_net}.txt"
#             file_path = os.path.join("test_time",name)
#             with open(file_path, 'r+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
#                 f.write(f"for index: {index} the time array is {time}")