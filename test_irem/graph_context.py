import torch
from graph_models import GraphEBM, GraphFC, GraphPonder, GraphRecurrent, GNNDTNet, GNNBlock, DT_recurrent
import torch.nn.functional as F
import os
# import pdb
from graph_dataset import Identity, ConnectedComponents, ShortestPath
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, SparseAdam, NAdam, RAdam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
# from easydict import EasyDict
import os.path as osp
from torch.nn.utils import clip_grad_norm
import numpy as np
# from imageio import imwrite
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import make_grid
import seaborn as sns
from torch_geometric.nn import global_mean_pool
import sys
from random import randrange
import gc


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')
parser.add_argument('--vary', action='store_true', help='vary size of graph')
parser.add_argument('--no_replay_buffer', action='store_true', help='utilize a replay buffer')

parser.add_argument('--dataset', default='identity', type=str, help='dataset to evaluate')
parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='default', type=str, help='name of experiments')

# training
parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
parser.add_argument('--batch_size', default=512, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha hyperparameter')
parser.add_argument('--prog', action='store_true', help='use progressive loss')
parser.add_argument('--json_name', default='default', type=str, help='name of json file losses are written to')
parser.add_argument('--plot', action='store_true', help='plot the testing loss in graph')
parser.add_argument('--plot_name', default=None, type=str, help='prefix of file plot is written to')
parser.add_argument('--plot_folder', default=None, type=str, help='folder of file plot is written to')
parser.add_argument('--transfer_learn', action='store_true', help='do transfer learning')
parser.add_argument('--transfer_learn_model', default=None, type=str, help='model path to learn from')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

# EBM specific settings
parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
parser.add_argument('--rank', default=10, type=int, help='rank of matrix to use')
parser.add_argument('--num_steps', default=5, type=int, help='Steps of gradient descent for training')
parser.add_argument('--step_lr', default=100.0, type=float, help='step size of latents')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--decoder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--gen', action='store_true', help='evaluate generalization')
parser.add_argument('--gen_rank', default=5, type=int, help='Add additional rank for generalization')
parser.add_argument('--recurrent', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--dt', action='store_true', help='use deep thinking net')
parser.add_argument('--ponder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--no_truncate_grad', action='store_true', help='not truncate gradient')
parser.add_argument('--iterative_decoder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--mem', action='store_true', help='add external memory to compute answers')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--capacity', default=50000, type=int, help='number of elements to generate')
parser.add_argument('--infinite', action='store_true', help='makes the dataset have an infinite number of elements')

def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def main_single(rank, FLAGS):

    if FLAGS.dataset == 'identity':
        dataset = Identity('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = Identity('test', FLAGS.rank)
        gen_dataset = Identity('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'connected':
        dataset = ConnectedComponents('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = ConnectedComponents('test', FLAGS.rank)
        gen_dataset = ConnectedComponents('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'shortestpath':
        dataset = ShortestPath('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = ShortestPath('test', FLAGS.rank)
        gen_dataset = ShortestPath('test', FLAGS.rank+FLAGS.gen_rank)

    # train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    gen_dataloader = DataLoader(gen_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size // 2, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    i=0
    stacked = []
    max_iters = 10000
    print("entering loop")
    for data in test_dataloader:
        if i < max_iters:
            # print(i)
            i +=1
            im = data['y']
            # print(type(torch.mean(im).item()))
            # sys.exit()
            stacked.append(torch.mean(im).item())
        else:
            break
        # if i==1:
        #     print("data shape is ",data['y'].shape)
        #     stacked = torch.squeeze(data['y'])
        #     continue
        # im = data['y']
        # # print(torch.mean(im))
        # stacked = torch.concat((stacked,torch.squeeze(im)))
        # if i == 1000:
        #     print("stacked shape is ",stacked.shape)
        #     print("mean is ",torch.mean(stacked))
        #     sys.exit()
    # print("mean is ", np.mean(np.array(stacked)))
    return np.mean(np.array(stacked))

def main():
    FLAGS = parser.parse_args()
    FLAGS.replay_buffer = not FLAGS.no_replay_buffer
    rank_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
    # print(rank_list)
    # sys.exit()
    averages_list = []
    for rank in rank_list:
        FLAGS.rank = rank
        print(rank)
        averages_list.append(main_single(0, FLAGS))
    print("saving")
    np.save("changes_array", np.array(averages_list))


# if __name__ == "__main__":
#     main()

arr = np.load("changes_array.npy")
f, ax = plt.subplots(1, 1)

# ax.plot(in_dist, label=f"{FLAGS.rank} nodes for {data} experiment")
# ax.plot(out_dist, label=f"{FLAGS.gen_rank+FLAGS.rank} nodes for {data} experiment")
ax.set(ylabel='Average Distance of Shortest Path', xlabel='Number of Nodes in graph')#, title="AA score for maze models over alpha values")
# ax1 = ax.twinx()
# ax1.set_ylim(ax.get_ylim())
# ax1.set_yticks([out_dist[out_dist.size-1], in_dist[in_dist.size-1]])
rank_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
axis = []
for num in rank_list:
    axis.append((num/5)-1)
strings = []
for num in rank_list:
    strings.append(str(num))
ax.plot(arr)
ax.set_xticks(axis, strings, rotation = (45), fontsize = 8)
# ax.set_xticklabels(rotation = (45), fontsize = 10, va='bottom', ha='left')
# ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
plt.savefig("context_graph", bbox_inches="tight", dpi=500)