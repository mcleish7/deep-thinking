import torch
from torch_geometric.nn import GINEConv, global_max_pool
from torch import nn
import torch.nn.functional as F
import torch_geometric
import sys

class GraphEBM(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(GraphEBM, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h // 2)
        self.edge_map_opt = nn.Linear(1, h // 2)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, inp, opt_edge):

        edge_embed = self.edge_map(inp.edge_attr)
        opt_edge_embed = self.edge_map_opt(opt_edge)

        edge_embed = torch.cat([edge_embed, opt_edge_embed], dim=-1)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        mean_feat = global_max_pool(h, inp.batch)
        energy = self.fc2(F.relu(self.fc1(mean_feat)))

        return energy


class GraphIterative(nn.Module):
    def __init__(self, inp_dim, out_dim, mem):
        super(GraphIterative, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h // 2)
        self.edge_map_opt = nn.Linear(1, h // 2)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, opt_edge):

        edge_embed = self.edge_map(inp.edge_attr)
        opt_edge_embed = self.edge_map_opt(opt_edge)

        edge_embed = torch.cat([edge_embed, opt_edge_embed], dim=-1)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)

        return output


class GraphRecurrent(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphRecurrent, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)

        self.lstm = nn.LSTM(h, h)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, state=None):
        # print("x shape is ", inp.x.shape)
        # print("edge index shape is ", inp.edge_index.shape)
        # print("y shape is ", inp.y.shape)
        edge_embed = self.edge_map(inp.edge_attr)
        
        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)

        h, state = self.lstm(h[None], state)
        h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)
        # print("h shape is ",h.shape)
        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]
        # print("h1 shape is ", h1.shape)
        h = torch.cat([h1, h2], dim=-1)
        # print("h shape is ", h.shape)
        output = self.decode(h)
        # print("output shape is ", output.shape)
        # sys.exit()
        return output, state


class GraphPonder(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphPonder, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, iters=1):

        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        outputs = []

        for i in range(iters):
            h = F.relu(self.conv2(h, inp.edge_index, edge_attr=edge_embed))

            output = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

            edge_index = inp.edge_index
            hidden = output

            h1 = hidden[edge_index[0]]
            h2 = hidden[edge_index[1]]

            output = torch.cat([h1, h2], dim=-1)
            output = self.decode(output)

            outputs.append(output)

        return outputs



class GraphFC(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(GraphFC, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)


    def forward(self, inp):

        edge_embed = self.edge_map(inp.edge_attr)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        h = self.conv3(h, inp.edge_index, edge_attr=edge_embed)

        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)

        return output


class GNNBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inp_dim, out_dim, stride=1):
        super(GNNBlock, self).__init__()
        self.conv1 = GINEConv(inp_dim, out_dim)
        self.conv2 = GINEConv(out_dim, out_dim)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inp_dim != self.expansion * out_dim:
            self.shortcut = torch_geometric.nn.Sequential(GINEConv(inp_dim, self.expansion * out_dim))
        # print("in block and the block is",self.conv1,self.conv2,self.shortcut)
    def forward(self, x, edge_index):
    # def forward(self, data, state = None):
        # x, edge_index = data.x, data.edge_index
        # print("in block foward")
        out = F.relu(self.conv1(x, edge_index))
        out = self.conv2(out, edge_index)
        out += self.shortcut(out)
        out = F.relu(out)
        # print("applied once")
        return out

class GNNBlockLinear(torch.nn.Module):
    expansion = 1
    def __init__(self, inp_dim, out_dim, stride=1):
        super(GNNBlockLinear, self).__init__()
        h=128
        self.conv1 = GINEConv(nn.Sequential(nn.Linear(inp_dim, out_dim)), edge_dim=h)
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(out_dim, out_dim)), edge_dim=h)

        self.shortcut = torch.nn.Sequential()

        if stride != 1 or inp_dim != self.expansion * out_dim:
            self.shortcut = torch_geometric.nn.Sequential(GINEConv(inp_dim, self.expansion * out_dim))

    def forward(self, x, edge_index, edge_embed):
        out = F.relu(self.conv1(x, edge_index, edge_attr=edge_embed))
        out = self.conv2(out, edge_index, edge_attr=edge_embed)
        out += self.shortcut(out)
        out = F.relu(out)
        # print("applied once")
        return out

class GNNDTNet(torch.nn.Module):
    def __init__(self, width, in_channels=3, recall=True, group_norm=False, **kwargs):
        block = GNNBlockLinear
        num_blocks = [2]

        super().__init__()

        self.recall = recall
        self.width = int(width)

        self.edge_map = nn.Linear(1, width)

        self.projection = GINEConv(nn.Sequential(nn.Linear(1, 128), nn.ReLU(inplace=True)), edge_dim=width)#GINEConv(in_channels, width, edge_dim=width)

        # conv_recall = GINEConv(width + in_channels, width)
        conv_recall = GINEConv(nn.Sequential(nn.Linear(width + in_channels, width)), edge_dim=width)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            # recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))
            recur_layers = recur_layers+self._make_layer(block, width, num_blocks[i], stride=1)

        head_conv1 = GINEConv(nn.Sequential(nn.Linear(width, 32)), edge_dim=width)#GINEConv(width, 32, edge_dim=width)
        head_conv2 = GINEConv(nn.Sequential(nn.Linear(32, 8)), edge_dim=width)#GINEConv(32, 8, edge_dim=width)
        head_conv3 = GINEConv(nn.Sequential(nn.Linear(8, 1)), edge_dim=width)#GINEConv(8, 2, edge_dim=width)

        # self.projection = torch_geometric.nn.Sequential(proj_conv, torch.nn.ReLU())
        # self.projection = torch_geometric.nn.Sequential('x, edge_index',[(self.proj_conv, 'x, edge_index -> x'), torch.nn.ReLU()])

        # print("recurlayers")
        # print(recur_layers)
        # print("changed")
        print(self.change_to_seq(recur_layers))
        # sys.exit()
        seq_recur_layers = self.change_to_seq(recur_layers)
        # self.recur_block = torch_geometric.nn.Sequential('x, edge_index',[*recur_layers])
        self.recur_block = torch_geometric.nn.Sequential(*seq_recur_layers)
        self.head = torch_geometric.nn.Sequential('x, edge_index, edge_attr ',[
                                    (head_conv1, 'x, edge_index, edge_attr  -> x'), torch.nn.ReLU(),
                                    (head_conv2, 'x, edge_index, edge_attr -> x'), torch.nn.ReLU(),
                                    (head_conv3, 'x, edge_index, edge_attr  -> x')])


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
            layers1.append((layers[i],'x, edge_index, edge_attr -> x'))
        return ['x, edge_index, edge_attr',layers1]

    def toedges(self, hidden, edge_index):
        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]
        # print("h1 shape is ", h1.shape)
        h = torch.cat([h1, h2], dim=-1)
        # print("h shape is ", h.shape)
        decode = nn.Linear(2, 1).to("cuda" if torch.cuda.is_available() else "cpu")
        return decode(h)
        # print("output shape is ", output.shape)

    def forward(self, data, iters_to_do=5, interim_thought=None, **kwargs):
        edge_embed = self.edge_map(data.edge_attr)

        # print("data y shape is ",data.y.shape)
        x, edge_index = data.x, data.edge_index
        # print("x size is ", x.shape)
        # print(x.size(0))
        # print("edge index size is ", edge_index.shape)
        # print(edge_index)
        # print(iters_to_do)
        # x =  self.GINE(x, edge_index).relu()
        # initial_thought = self.projection(x, edge_index, edge_attr = edge_embed)

        # initial_thought = self.projection(x, edge_index, edge_attr = edge_embed)
        initial_thought = self.projection(x, edge_index, edge_attr = edge_embed)
        # print("initla thought is ", initial_thought)
        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = [] # torch.zeros((x.size(0), iters_to_do, edge_index.size(0))).to(x.device)

        for i in range(iters_to_do):
            # print("on ",i)
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            # print("type of edge embed is ", type(edge_embed))
            interim_thought = self.recur_block(interim_thought, edge_index, edge_attr = edge_embed)
        
            # print("interim thought shape is", interim_thought.shape)
            # print(self.toedges(interim_thought, edge_index))
            out = self.head(interim_thought, edge_index, edge_attr = edge_embed)

            # print("out shape is ",out.shape)

            all_outputs.append(self.toedges(out, edge_index))

        # print("passed loop")
        # print("out shape is ",out.shape)
        # output = self.toedges(out, edge_index)

        # print("output shape is ", output.shape)
        

        preds = torch.stack(all_outputs, dim=1)
        # print("preds shape is ",preds.shape)
        # print("final ouput is ",out)
        # print("first output is ",all_outputs[0])
        # print("y is ",data.y)
        # sys.exit()
        return all_outputs, interim_thought
        # return all_outputs