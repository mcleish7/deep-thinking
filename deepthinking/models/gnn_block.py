import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

class GNNBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inp_dim, out_dim, stride=1):
        super(GNNBlock, self).__init__()
        self.conv1 = GCNConv(inp_dim, out_dim)
        self.conv2 = GCNConv(out_dim, out_dim)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inp_dim != self.expansion * out_dim:
            self.shortcut = torch_geometric.nn.Sequential(GCNConv(inp_dim, self.expansion * out_dim))
        # print("in block and the block is",self.conv1,self.conv2,self.shortcut)
    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        # print("in block foward")
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        out += self.shortcut(out)
        out = F.relu(out)
        # print("applied once")
        return out

# class BasicBlock2D(nn.Module):
#     """Basic residual block class 2D"""

#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, group_norm=False):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
#                                                     kernel_size=1, stride=stride, bias=False))

#     def forward(self, x):
#         out = F.relu(self.gn1(self.conv1(x)))
#         out = self.gn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class GraphRecurrent(nn.Module):
#     def __init__(self, inp_dim, out_dim):
#         super(GraphRecurrent, self).__init__()
#         h = 128

#         self.edge_map = nn.Linear(1, h)

#         self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)

#         self.lstm = nn.LSTM(h, h)
#         # self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
#         self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

#         self.decode = nn.Linear(256, 1)

#     def forward(self, inp, state=None):

#         edge_embed = self.edge_map(inp.edge_attr)

#         h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)

#         h, state = self.lstm(h[None], state)
#         h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)

#         edge_index = inp.edge_index
#         hidden = h

#         h1 = hidden[edge_index[0]]
#         h2 = hidden[edge_index[1]]

#         h = torch.cat([h1, h2], dim=-1)
#         output = self.decode(h)

#         return output, state