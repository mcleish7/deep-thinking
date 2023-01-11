import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric
from .gnn_block import GNNBlock
import sys

class GNNDTNet(torch.nn.Module):
    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, group_norm=False, **kwargs):
        super().__init__()

        self.recall = recall
        self.width = int(width)

        proj_conv = GCNConv(in_channels, width)

        conv_recall = GCNConv(width + in_channels, width)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            # recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))
            recur_layers = recur_layers+self._make_layer(block, width, num_blocks[i], stride=1)

        head_conv1 = GCNConv(width, 32)
        head_conv2 = GCNConv(32, 8)
        head_conv3 = GCNConv(8, 2)

        # self.projection = torch_geometric.nn.Sequential(proj_conv, torch.nn.ReLU())
        self.projection = torch_geometric.nn.Sequential('x, edge_index',[(proj_conv, 'x, edge_index -> x'), torch.nn.ReLU()])
        # print("recurlayers")
        # print(recur_layers)
        # print("cjhanged")
        # print(self.change_to_seq(recur_layers))
        # sys.exit()
        seq_recur_layers = self.change_to_seq(recur_layers)
        # self.recur_block = torch_geometric.nn.Sequential('x, edge_index',[*recur_layers])
        self.recur_block = torch_geometric.nn.Sequential(*seq_recur_layers)
        self.head = torch_geometric.nn.Sequential('x, edge_index',[
                                    (head_conv1, 'x, edge_index -> x'), torch.nn.ReLU(),
                                    (head_conv2, 'x, edge_index -> x'), torch.nn.ReLU(),
                                    (head_conv3, 'x, edge_index -> x')])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd))
            self.width = planes * block.expansion
        # print("layers")
        # print(layers)
        # print(type(layers))
        # print(layers[0])
        # print(type(layers[0]))
        # print("end layers")
        # layers1 = []
        # for i in range(0,len(layers)-1):
        #     layers1.append((layers[i],'x, edge_index -> x'))
        # layers1.append(layers[len(layers)-1])
        l = self.change_to_seq(layers)
        # print("l")
        # print(l)
        # print("end l")
        # return torch_geometric.nn.Sequential(*l)
        return layers
    def change_to_seq(self,layers):
        layers1 = []
        # for i in range(0,len(layers)-1):
        #     layers1.append((layers[i],'x, edge_index -> x'))
        # layers1.append(layers[len(layers)-1])
        for i in range(0,len(layers)):
            layers1.append((layers[i],'x, edge_index -> x'))
        return ['x, edge_index',layers1]

    def forward(self, data, iters_to_do=5, interim_thought=None, **kwargs):
        # print("in forward")
        x, edge_index = data.x, data.edge_index
        # print("x size is ", x.shape)
        # print("edge index size is ", edge_index.shape)
        # x =  self.gcn(x, edge_index).relu()
        initial_thought = self.projection(x, edge_index)
        # initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0),iters_to_do, edge_index.size(0))).to(x.device)

        for i in range(iters_to_do):
            # print("on ",i)
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)

            interim_thought = self.recur_block(interim_thought, edge_index)
            out = self.head(interim_thought, edge_index)
            print("out shape is ",out.shape)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought
        return out
        # return all_outputs

class GraphRecurrent(torch.nn.Module):
    # Taken from https://github.com/yilundu/irem_code_release/blob/d0de28d2dc08a255c5c387d207c155b820cd6a15/graph_models.py#L76
    def __init__(self):
        super(GraphRecurrent, self).__init__()
        h = 128

        self.edge_map = torch.nn.Linear(1, h)

        self.conv1 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(1, 128)), edge_dim=h)

        self.lstm = torch.nn.LSTM(h, h)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = torch_geometric.nn.GINEConv(torch.nn.Sequential(torch.nn.Linear(128, 128)), edge_dim=h)

        self.decode = torch.nn.Linear(256, 1)

    def forward(self, inp, iters_to_do=1000, interim_thought=None, state=None):
        # iters = 20
        # for i in range(0,iters):
        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)

        h, state = self.lstm(h[None], state)
        h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]
        # print("h1 is shape ", h1.shape)
        # print("h2 is shape ", h2.shape)
        h = torch.cat([h1, h2], dim=-1)
        # print("h is ", h.shape)
        output = self.decode(h)
        # print("output is ", output.shape)
        return output, state

# class DTNet(nn.Module):
#     """DeepThinking Network 2D model class"""

#     def __init__(self, block, num_blocks, width, in_channels=3, recall=True, group_norm=False, **kwargs):
#         super().__init__()

#         self.recall = recall
#         self.width = int(width)
#         self.group_norm = group_norm
#         proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
#                               stride=1, padding=1, bias=False)

#         conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
#                                 stride=1, padding=1, bias=False)

#         recur_layers = []
#         if recall:
#             recur_layers.append(conv_recall)

#         for i in range(len(num_blocks)):
#             recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

#         head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
#                                stride=1, padding=1, bias=False)

#         self.projection = nn.Sequential(proj_conv, nn.ReLU())
#         self.recur_block = nn.Sequential(*recur_layers)
#         self.head = nn.Sequential(head_conv1, nn.ReLU(),
#                                   head_conv2, nn.ReLU(),
#                                   head_conv3)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for strd in strides:
#             layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
#             self.width = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, iters_to_do=1000, interim_thought=None, **kwargs):
#         initial_thought = self.projection(x)

#         if interim_thought is None:
#             interim_thought = initial_thought

#         all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

#         for i in range(iters_to_do):
#             if self.recall:
#                 interim_thought = torch.cat([interim_thought, x], 1)
#             interim_thought = self.recur_block(interim_thought)
#             out = self.head(interim_thought)
#             all_outputs[:, i] = out

#         if self.training:
#             return out, interim_thought

#         return all_outputs

def dt_gnn_recall(width, **kwargs):
    # return GNNDTNet(GNNBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)
    return GraphRecurrent()

# Testing
if __name__ == "__main__":
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1],
                            [1, 0],
                            [1, 2],
                            [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())
    net = dt_gnn_recall(width=5, max_iters=5, in_channels=1)
    print(net)
    # print(summary(net))
    # x_test = torch.rand(30).reshape([3, 1, 10])
    print(type(data))
    print(data)
    out_test, _ = net(data)
    print("complete run")
    # print(out_test.shape)
    # out_test, _ = net(x_test, n=2, k=2)
    # print(out_test.shape)

    net.eval()
    # outputs = net(x_test)
    # print(outputs.shape)
    # outputs = net(x_test, n=2, k=2)
    # print(outputs.shape)