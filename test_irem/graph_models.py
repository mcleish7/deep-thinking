import torch
from torch_geometric.nn import GINEConv, global_max_pool, GATConv, GMMConv, GENConv, PDNConv, SAGEConv, EdgeConv
from torch import nn
import torch.nn.functional as F
import torch_geometric
import sys

sys.path.append('/dcs/large/u2004277/deep-thinking')
import deepthinking.models as dt_models

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
        # print(torch.cuda.is_available())
        # print(inp.batch.shape) #5120
        # print("before shape is ", h.shape) #torch.Size([5120, 128])
        mean_feat = global_max_pool(h, inp.batch)
        # print("before shape is ", mean_feat.shape) #torch.Size([512, 128])
        # sys.exit()
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
        # return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)

        self.decode = nn.Linear(256, 1)

    def forward(self, inp, state=None):
        print("x shape is ", inp.x.shape)
        print("edge index shape is ", inp.edge_index.shape)
        print("edge attr shape is ", inp.edge_attr.shape)
        print("y shape is ", inp.y.shape)

        edge_embed = self.edge_map(inp.edge_attr)
        print("edge embed shape is ",edge_embed.shape)
        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        print("h1 shape is ",h.shape)
        h, state = self.lstm(h[None], state)
        # print(state)
        print("state is shape ", state[0].shape)
        print("h2 shape is ",h.shape)
        h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)
        print("h3 shape is ",h.shape)
        edge_index = inp.edge_index
        hidden = h

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]
        print("h1 shape is ", h1.shape)
        h = torch.cat([h1, h2], dim=-1)
        print("h shape is ", h.shape)
        output = self.decode(h)
        print("output shape is ", output.shape)
        sys.exit()
        return output, state

class DT_recurrent(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DT_recurrent, self).__init__()
        h = 128

        self.edge_map = nn.Linear(1, h)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=h)
        # self.conv11 = GINEConv(nn.Sequential(nn.Linear(256, 256)), edge_dim=h)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(256, h)), edge_dim=h)#EdgeConv(nn.Sequential(nn.Linear(128, 128)))#SAGEConv(h,h)#GINEConv(nn.Sequential(nn.Linear(128, 128)), edge_dim=h)
        # self.lstm = nn.LSTM(h, h)
        self.recur = dt_models.dt_net_recall_1d(width = h, in_channels = 128, graph=True)
        # print(self.recur)
        # sys.exit()
        # return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)
        # self.2d_1d = nn.conv
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(h, h)), edge_dim=h)

        self.decode = nn.Linear((2*h), 1)

        self.dropout = nn.Dropout(p=0.25, inplace=False)

        # self.gn = torch_geometric.nn.norm.batch_norm.BatchNorm(h)
        # self.relu = nn.PReLU(num_parameters=128)
        # self.relu1 = nn.PReLU(num_parameters=1)
        # self.decode2 = nn.Linear(128, 1)  

    def forward(self, inp, iters_to_do=10, iters_elapsed=0, interim_thought=None, **kwargs):
        # print("x shape is ", inp.x.shape)
        # print("edge index shape is ", inp.edge_index.shape)
        # print("edge attr shape is ", inp.edge_attr.shape)
        # print("y shape is ", inp.y.shape)
        all_outputs = []
        mean, std, var = torch.mean(inp.edge_attr), torch.std(inp.edge_attr), torch.var(inp.edge_attr)
        edge_attr = (inp.edge_attr-mean)/std

        edge_embed = self.edge_map(edge_attr)
        # print("edge embed shape is ",edge_embed.shape)

        h = self.conv1(inp.x, inp.edge_index, edge_attr=edge_embed)
        # h = self.relu(h)
        # print("h shape is ", h.shape)
        # print("edge index shape is ", inp.edge_index.shape)
        # h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
        # print("h1 shape is ",h.shape)
        # h = h.T#torch.unsqueeze(h.T,0)
        # print("h[none] shape is ", h[None].shape)
        initial_h = h.T
        for i in range(0, iters_to_do):
            # if i>0:
            #     interim_thought = self.dropout(interim_thought)
            _, interim_thought = self.recur(initial_h[None], interim_thought=interim_thought, iters_to_do=1)
            # interim_thought = self.relu1(interim_thought)
            h = torch.unsqueeze(torch.squeeze(interim_thought).T,0)
            # print(state)
            # print("state is shape ", state[0].shape)
            # print("h2 shape is ",h.shape)
            # h = self.dropout(h)
            # h = self.relu(h)
            h = self.conv3(h[0], inp.edge_index, edge_attr=edge_embed)
            # h = self.gn(h)
            
            # h = self.conv2(h, inp.edge_index, edge_attr=edge_embed)
            # print("h3 shape is ",h.shape)
            edge_index = inp.edge_index
            hidden = h

            h1 = hidden[edge_index[0]]
            h2 = hidden[edge_index[1]]
            # print("h1 shape is ", h1.shape)
            h = torch.cat([h1, h2], dim=-1)
            # print("h shape is ", h.shape)
            output = self.decode(h)
            all_outputs.append(output)
        # print("output shape is ", output.shape)
        # sys.exit()
        return all_outputs, interim_thought

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
        # self.conv1 = GINEConv(nn.Sequential(nn.Linear(inp_dim, out_dim)), edge_dim=h)
        # self.conv2 = GINEConv(nn.Sequential(nn.Linear(out_dim, out_dim)), edge_dim=h)
        self.conv1 = PDNConv(inp_dim, out_dim, edge_dim=h, hidden_channels = 128)
        self.conv2 = PDNConv(inp_dim, out_dim, edge_dim=h, hidden_channels = 128)

        self.shortcut = torch.nn.Sequential()

        if stride != 1 or inp_dim != self.expansion * out_dim:
            print("made shortcut")
            self.shortcut = torch_geometric.nn.Sequential(GINEConv(inp_dim, self.expansion * out_dim))

    def forward(self, x, edge_index, edge_embed):

        out = F.relu(self.conv1(x, edge_index, edge_attr=edge_embed))
        # out = self.conv1(x, edge_index, edge_attr=edge_embed)
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

        self.projection = GINEConv(nn.Sequential(nn.Linear(1, self.width), nn.ReLU(inplace=True)), edge_dim=width)#GINEConv(in_channels, width, edge_dim=width)
        # self.projection = GINEConv(nn.Sequential(nn.Linear(1, 128)), edge_dim=width)#GINEConv(in_channels, width, edge_dim=width)

        # conv_recall = GINEConv(width + in_channels, width)
        self.conv_recall = GINEConv(nn.Sequential(nn.Linear(width + in_channels, width)), edge_dim=width)

        recur_layers = []
        if recall:
            recur_layers.append(self.conv_recall)

        for i in range(len(num_blocks)):
            # recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))
            recur_layers = recur_layers+self._make_layer(block, width, num_blocks[i], stride=1)

        self.head_conv0 = GINEConv(nn.Sequential(nn.Linear(width, 128)), edge_dim=width)
        # # self.head_conv01 = GINEConv(nn.Sequential(nn.Linear(128, 64)), edge_dim=width)
        self.head_conv1 = GINEConv(nn.Sequential(nn.Linear(128, 32)), edge_dim=width)#GINEConv(width, 32, edge_dim=width)
        self.head_conv2 = GINEConv(nn.Sequential(nn.Linear(32, 8)), edge_dim=width)#GINEConv(32, 8, edge_dim=width)
        self.head_conv3 = GINEConv(nn.Sequential(nn.Linear(8, 1)), edge_dim=width)#GINEConv(8, 2, edge_dim=width)

        # self.head_conv0 = PDNConv(width, 128, edge_dim=width, hidden_channels = 128)
        # self.head_conv1 = PDNConv(128, 32, edge_dim=width, hidden_channels = 64)
        # self.head_conv2 = PDNConv(32, 8, edge_dim=width, hidden_channels = 32)
        # self.head_conv3 = PDNConv(8, 1, edge_dim=width, hidden_channels = 8)

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
                                    (self.head_conv1, 'x, edge_index, edge_attr  -> x'), torch.nn.ReLU(inplace=True),
                                    (self.head_conv2, 'x, edge_index, edge_attr -> x'), torch.nn.ReLU(inplace=True),
                                    (self.head_conv3, 'x, edge_index, edge_attr  -> x')])

        self.middle_1 = GNNBlockLinear(self.width, self.width)
        self.middle_2 = GNNBlockLinear(self.width, self.width)
        self.middle_3 = GNNBlockLinear(self.width, self.width)
        self.middle_4 = GNNBlockLinear(self.width, self.width)
        self.gn = torch_geometric.nn.norm.batch_norm.BatchNorm(self.width)
        self.gn1 = torch_geometric.nn.norm.batch_norm.BatchNorm(self.width)
        self.gn2 = torch_geometric.nn.norm.batch_norm.BatchNorm(self.width)
        self.gn3 = torch_geometric.nn.norm.batch_norm.BatchNorm(self.width)
        self.gn4 = torch_geometric.nn.norm.batch_norm.BatchNorm(128)
        # self.gn5 = torch_geometric.nn.norm.msg_norm.MessageNorm(64)
        self.gn5 = torch_geometric.nn.norm.batch_norm.BatchNorm(32)
        self.gn6 = torch_geometric.nn.norm.batch_norm.BatchNorm(8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # print("strides is ", strides)
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

    def forward(self, data, iters_to_do=5, iters_elapsed=0, interim_thought=None, **kwargs):
        # iters_to_do = iters_to_do - iters_elapsed

        edge_embed = self.edge_map(data.edge_attr)
        edge_embed = self.gn1(edge_embed)
        # print(edge_embed.shape)
        # sys.exit()
        # print("data y shape is ",data.y.shape)
        x, edge_index = data.x, data.edge_index#.float()
        # print("x size is ", x.shape)
        # print(x)
        # x = self.gn4(x)
        # print(x)
        # print("x size is ", x.shape)
        # print("edge index shape is ", edge_index.shape)
        # print(edge_index)
        # edge_index = self.gn5(edge_index.T).T.int()
        # print("edge index shape is ", edge_index.shape)
        # print(edge_index)
        # sys.exit()

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
        # print("x shape is ", x.shape)
        # print("initial thought shape is ", initial_thought.shape)
        
        all_outputs = [] # torch.zeros((x.size(0), iters_to_do, edge_index.size(0))).to(x.device)

        for i in range(iters_to_do):
            # print("on ",i)
            
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
                # print("interim thought is shape ", interim_thought.shape)
                # sys.exit()
            # print("type of edge embed is ", type(edge_embed))

            # print("interim thought 1 is ", interim_thought.shape)
            # interim_thought = self.recur_block(interim_thought, edge_index, edge_attr = edge_embed)

            interim_thought = self.conv_recall(interim_thought, edge_index, edge_attr = edge_embed)
            interim_thought = self.middle_1(interim_thought, edge_index, edge_embed = edge_embed)
            interim_thought = self.gn(interim_thought)
            interim_thought = self.middle_2(interim_thought, edge_index, edge_embed = edge_embed)
            interim_thought = self.gn2(interim_thought)
            interim_thought = self.middle_3(interim_thought, edge_index, edge_embed = edge_embed)
            interim_thought = self.gn3(interim_thought)
            interim_thought = self.middle_4(interim_thought, edge_index, edge_embed = edge_embed)


            # interim_thought = self.gn(interim_thought)

            # interim_thought = self.middle(interim_thought, edge_index, edge_embed = edge_embed)
            # interim_thought = self.middle(interim_thought, edge_index, edge_embed = edge_embed)
            # interim_thought = self.middle(interim_thought, edge_index, edge_embed = edge_embed)

            # print("interim thought shape is", interim_thought.shape)
            # sys.exit()
            # print(self.toedges(interim_thought, edge_index))
            # print("before head is shape ", interim_thought.shape)
            out = F.relu(self.head_conv0(interim_thought, edge_index, edge_attr = edge_embed))
            # out1 = self.gn4(out1, interim_thought)
            # out2 = F.relu(self.head_conv01(out1, edge_index, edge_attr = edge_embed))
            out = self.gn4(out)
            out = F.relu(self.head_conv1(out, edge_index, edge_attr = edge_embed))
            # print("out shape is ", out.shape)
            out = self.gn5(out)
            out = F.relu(self.head_conv2(out, edge_index, edge_attr = edge_embed))
            out = self.gn6(out)
            out = F.relu(self.head_conv3(out, edge_index, edge_attr = edge_embed))
            # interim_thought = out
            # print("interim thought 2 shape is ", interim_thought.shape)
            # sys.exit()
            # out = self.head_one(interim_thought, edge_index, edge_attr = edge_embed)

            # print("out shape is ",out.shape)

            all_outputs.append(self.toedges(out, edge_index))
        #     print(" at iteration ",i," out is ", out)
        # print("data y is ", data.y)
        # sys.exit()
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