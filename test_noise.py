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
import string

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

# net = getattr(models, "dt_net_1d")(width=400, in_channels=3, max_iters=30) # for Verena
# state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-frockless-Verena/model_best.pth", map_location=device)

net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30) # for Betz => recall, alpha =0
state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-peeling-Betzaida/model_best.pth", map_location=device)

# net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300) # for Jojo => recall, alpha =1
# state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo/model_best.pth", map_location=device)

#print(type(net))
#net = dt.get_model("dt_net_recall_1d", 400, 30)
net = net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(state_dict["net"])

net.eval()

ex = torch.zeros((3, 1, 400), dtype=torch.float)


# a = [[[ 0., -1.,  0., -1.,  0., -1., -1., -1.,  0., -1.,  0.,  0., -1.,  0.,-1., -1.,  0.,  0., -1., -1.,  0.,  0.,  0.,  0., -1.,  0., -1.,  0.,-1., -1., -1., -1.]]]
# t =[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1,1, 0, 0, 1, 1, 1, 1, 1]

# a = np.array(a, dtype = float)
# a = torch.from_numpy(a)
# input = a.to(device, dtype=torch.float)
# t = np.array(t, dtype = float)
# t = torch.from_numpy(t)
# target = t.to(device, dtype=torch.float)
def get_data():
    data = torch.load("batch_shells_sums/data/prefix_sums_data/48_data.pth").unsqueeze(1) - 0.5
    target = torch.load("batch_shells_sums/data/prefix_sums_data/48_targets.pth")
    a = data[1]
    input = a.to(device, dtype=torch.float).unsqueeze(0) #to account for batching in real net

    t = target[1]
    t = t.to(device, dtype=torch.float)#.long()
    target = t.unsqueeze(0)
    return input, target
input, target = get_data()
# input = torch.load("batch_shells_sums/data/prefix_sums_data/56_data.pth")[0]
# input = input.view(ex.size(), -1)
# #print(data)
# target = torch.load("batch_shells_sums/data/prefix_sums_data/56_targets.pth")[0]
#print(target)
# print("real input is"+str(input))
iters =300
corrects = torch.zeros(iters)
# print("input shape is ",input.shape)
with torch.no_grad():
    output = net(input,iters_to_do=iters) #iters_to_do=50
# print("real output ßis"+str(output))
# print("output shape is "+str(output.shape))
outputs_max_iters = output.clone().view(output.size(0),output.size(1), -1)

# print("testing augementation has shape",outputs_max_iters.shape)
# golden_label = list(torch.argmax(output, dim=3))
for i in range(output.size(1)):
    outputi = output[:, i]
    predicted = outputi.clone().argmax(1)
    # print("predicted shape is "+str(predicted.shape))#should take away 2 dimensions added by ne
    golden_label = predicted.view(predicted.size(0), -1)
    # print("predicted shape is "+str(predicted.shape))
    #.item()
    # print("Error-free label:", golden_label.shape)
    # print("Error-free label:"+str(golden_label))
    corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 

correct = corrects.cpu().detach().numpy()
bestind = np.argmax(correct)
print("best is",bestind)
best = output[:,bestind]
predicted = best.clone().argmax(1)
golden_label = predicted.view(predicted.size(0), -1)
print("Best Error-free label:"+str(golden_label))
print("has accuracy ", correct[bestind]," out of 32")

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

t1,t2 = net_out_to_bits(output,target)
print("predicted string is ",t1)
print("number of matches is ",t2)


batch_size = 500
channels = 1
width = 400
height = width
layer_types_input = [torch.nn.Conv1d]
#print(next(net.parameters()).is_cuda)
pfi_model = fault_injection(net, 
                            batch_size,
                            input_shape=[channels,width],
                            #input_shape=[channels,height,width],
                            layer_types=layer_types_input,
                            use_cuda=True
                            )
#print(pfi_model.print_pytorchfi_layer_summary())
b = [0]
layer = [4]
C = [1]
H = [16]
W = [400]
err_val = [1]
inj = pfi_model.declare_neuron_fi(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val) #dim 3 ignored for sums
#input.to(device)
# print(type(input))
inj_output = inj(input)
# print(inj_output)
# print(type(inj_output))
#inj_output.to("cpu")
#inj_output = torch.from_numpy(numpy.asarray(inj_output))
#torch.from_numpy(

inj_label,matches = net_out_to_bits(inj_output,target)
#.item()
print("[Single Error] PytorchFI label:", inj_label)
print("matches in inj_label is ",matches)

class custom_func(fault_injection):
    #used = False
    count = 0
    j = 0 
    def __init__(self, in_j,model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)
        self.j = in_j

    # define your own function
    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        # # print(input)
        # # print(output)
        # self.count += 1
        # # if self.count == 7:
        # #     sys.exit()
        # if self.count == 15:
        #     print("BEFORE:")
        #     print(output)
        #     print(len(output))
        #     print(type(output))
        #     print(output[0])
        #     print(type(output[0]))
        #     a = output[:][0]
        #     print(a.shape)
        #     # output[:] = 0.0 if input[:] < 1.0 else 1.0
        #     for i in range(0,a.size(0)):
        #         for j in range(0,a.size(1)):
        #             if a[i][j]<0.0:
        #                 a[i][j] = 0.0
        #             else:
        #                 a[i][j] = 1.0
        #     output = (a,)
        #     print("after:")
        #     print(output)
        #     print(type(output))
        # self.count +=1
        # if self.count == 3:
        #     print("triggered!!!!!!!!")
        #     output[:] = output*-1
        # print("shape is ",output.shape," at layer ",self.get_current_layer()) #is nice shape at all multiples of 8 need to call conevert to bits at this stage
        # if (self.get_current_layer() == 400):
        #     print("output at start of 400 is ",output)
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400):
            j = self.j #between 0 and 48
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -100.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,j] = 100.0

        # if (self.get_current_layer() == 400):
        #     print("output at end of 400 is ",output)
        # if (self.get_current_layer() == 408):
        #     print("output at 408 is ",output)

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            # print("max layers is ",self.get_total_layers())
            self.reset_current_layer()

with torch.no_grad():
    pfi_model_2 = custom_func(0,net, 
                            batch_size,
                            input_shape=[channels,width],
                            layer_types=layer_types_input,
                            use_cuda=True
                        )

    inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)

    inj_output = inj(input)
    print("inj output shape is ",inj_output[0,49].unsqueeze(0).shape)
    print("inj output changed is ", inj_output[0,49,0,0])
    print("inj changed approx is ", convert_to_bits(inj_output[0,49].unsqueeze(0)))
    inj_label,matches = net_out_to_bits(inj_output,target, log=False, graph=True)
    print("[Single Error] PytorchFI label from class:", inj_label)
    print("matches in inj_label is ",matches)

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
    save_path = os.path.join("test_noise_outputs","test_time_correctness.png")
    plt.savefig(save_path)

print("now going into loop")
with torch.no_grad():
    time = []
    for i in range(0,10):
        pfi_model_2 = custom_func(i,net, 
                                batch_size,
                                input_shape=[channels,width],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)
        inj_output = inj(input)
        time.append(count_to_correct(inj_output,target))
    print(time)
    graph_time(time)

