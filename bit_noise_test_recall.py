import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import deepthinking.models as models
import numpy

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

state_dict = torch.load("batch_shells_sums/outputs/prefix_sums_ablation/training-enraged-Jojo/model_best.pth", map_location=device)

net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30)
#print(type(net))
#net = dt.get_model("dt_net_recall_1d", 400, 30)
net = net.to(device)
net = torch.nn.DataParallel(net)
net.load_state_dict(state_dict["net"])

net.eval()
input = torch.zeros((3, 1, 400), dtype=torch.float)
output = net(input)
golden_label = list(torch.argmax(output, dim=1))[0]
#.item()
print("Error-free label:", golden_label)

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
b = [1]
layer = [4]
C = [1]
H = [399]
W = [400]
err_val = [1]
inj = pfi_model.declare_neuron_fi(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val)
#input.to(device)
# print(type(input))
inj_output = inj(input)
print(inj_output)
print(type(inj_output))
#inj_output.to("cpu")
#inj_output = torch.from_numpy(numpy.asarray(inj_output))
#torch.from_numpy(
inj_label = list(torch.argmax(inj_output, dim=1))[0]
#.item()
print("[Single Error] PytorchFI label:", inj_label)

class custom_func(fault_injection):
    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    # define your own function
    def mul_neg_one(self, module, input, output):
        output[:] = 1.0 if input == 0.0 else 0.0

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

pfi_model_2 = custom_func(net, 
                        batch_size,
                        input_shape=[channels,width],
                        layer_types=layer_types_input,
                        use_cuda=True
                     )

inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.mul_neg_one)

inj_output = inj(input)
inj_label = list(torch.argmax(inj_output, dim=1))[0]
#.item()
print("[Single Error] PytorchFI label:", inj_label)
