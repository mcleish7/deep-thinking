import torch
import numpy as np
import deepthinking.models as models
import random

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

def run(net, input, target, max_iters):  
    with torch.no_grad():
        corrects = torch.zeros(max_iters)
        all_outputs = net(input, iters_to_do=max_iters)
        for i in range(all_outputs.size(1)):
            output = all_outputs[:, i]

            # predicted = get_predicted(input, output)

            # corrects[i] += torch.amin(predicted == target, dim=[1]).sum().item()

def main_module(alpha, number, input_size, max_iters):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    averages = []
    inputs, targets, a = get_data(n=number, size = input_size)
    store = []
    net = get_net(device, alpha)
    for i in range(0,number):
        arun = run(net, inputs[i],targets[i], max_iters)
        store.append(arun)
    push = random.randint(0,number)
    for i in range(0,number):
        start = (i + push)%number
        arun = run(net, inputs[i],targets[i], max_iters)

def main():
    nets_list = [-1,1,2,3,4,5,6,7,8,9]


device = "cuda" if torch.cuda.is_available() else "cpu"
name = "heating-Mihcael"
full_path = f"mismatch/outputs/mazes_ablation/training-{name}/model_best.pth"
state_dict = torch.load(full_path, map_location=device)


net1 = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50)
net1 = net1.to(device)
net1 = torch.nn.DataParallel(net1)
net1.load_state_dict(state_dict["net"])
net1.eval()
# print(vars(net1))
# print("type is ",type(state_dict["net"]))
# print(state_dict["net"].keys())
state_dict = torch.load(full_path, map_location=device)
net2 = getattr(models, "dt_net_recall_2d_start")(width=128, in_channels=3, max_iters=50, start = torch.rand([1, 3, 32, 32]).to(device))
net2 = net2.to(device)
# print(vars(net2))
# print(net2.training)
# print(net2.show_start())
net2 = torch.nn.DataParallel(net2)
net2.load_state_dict(state_dict["net"], strict=False)
net2.eval()
# print("\n........................................................\n")
# print(vars(net2))
# print(type(net2._parameters))
# print(net2._parameters.keys())
# print(type(net1))
# print(net1.show_start())

# print(net2.show_start())

input, target, a = get_data(n=1)
print("input shape is ",input.shape)
print("target shape in",target.shape)
net = get_net(device, -1) 
all_outputs1 = net1(input)
all_outputs2 = net2(input)
print("output shape is ",all_outputs1.shape)

# for alpha: 8 the time array is [0.9971649646759033, 0.9980168342590332, 0.9968770146369934, 0.931258499622345, 0.9408687949180603, 0.7856437563896179, 0.9429271817207336, 0.9065360426902771, 0.8678450584411621]