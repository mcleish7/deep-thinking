import torch
from graph_models import GraphEBM, GraphFC, GraphPonder, GraphRecurrent, GNNDTNet, GNNBlock
import torch.nn.functional as F
import os
# import pdb
from graph_dataset import Identity, ConnectedComponents, ShortestPath
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, SparseAdam
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


def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, inputs):
        batch_size = len(inputs)
        if self._next_idx >= len(self._storage):
            self._storage.extend(inputs)
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = inputs
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = inputs[:split_idx]
                self._storage[:batch_size - split_idx] = inputs[split_idx:]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def _encode_sample(self, idxes):
        datas = []
        for i in idxes:
            data = self._storage[i]
            datas.append(data)

        return datas

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes), torch.Tensor(idxes)

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]


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

best_test_error = 10.0
best_gen_test_error = 10.0


def average_gradients(model):
    size = float(dist.get_world_size())

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def gen_answer(inp, FLAGS, model, pred, scratch, num_steps, create_graph=True):
    preds = []
    im_grads = []
    energies = []
    im_sups = []

    if FLAGS.decoder:
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.recurrent:
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        for i in range(num_steps):
            pred, state = model.forward(inp, state)
            preds.append(pred)
    elif FLAGS.dt:
        # print(model)
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        preds, state = model.forward(inp, iters_to_do = num_steps, iterim_though = None)
    elif FLAGS.ponder:
        preds = model.forward(inp, iters=num_steps)
        pred = preds[-1]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
    elif FLAGS.iterative_decoder:
        for i in range(num_steps):
            energy = torch.zeros(1)

            out_dim = model.out_dim

            im_merge = torch.cat([pred, inp], dim=-1)
            pred = model.forward(im_merge) + pred

            energies.append(torch.zeros(1))
            im_grads.append(torch.zeros(1))
    else:
        with torch.enable_grad():
            pred.requires_grad_(requires_grad=True)
            s = inp.size()
            scratch.requires_grad_(requires_grad=True)

            for i in range(num_steps):

                energy = model.forward(inp, pred)

                if FLAGS.no_truncate_grad:
                    im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=create_graph)
                else:
                    if i == (num_steps - 1):
                        im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=True)
                    else:
                        im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=False)
                pred = pred - FLAGS.step_lr * im_grad

                preds.append(pred)
                energies.append(energy)
                im_grads.append(im_grad)

    return pred, preds, im_grads, energies, scratch


def ema_model(model, model_ema, mu=0.999):
    for (model, model_ema) in zip(model, model_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(model):
    size = float(dist.get_world_size())

    for model in model:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(FLAGS, device, dataset):
    # if FLAGS.decoder:
    #     model = GraphFC(dataset.inp_dim, dataset.out_dim)
    # elif FLAGS.ponder:
    #     model = GraphPonder(dataset.inp_dim, dataset.out_dim)
    # elif FLAGS.recurrent:
    #     model = GraphRecurrent(dataset.inp_dim, dataset.out_dim)
    # elif FLAGS.iterative_decoder:
    #     model = IterativeFC(dataset.inp_dim, dataset.out_dim, FLAGS.mem)
    # else:
    #     model = GraphEBM(dataset.inp_dim, dataset.out_dim, FLAGS.mem)
    if FLAGS.recurrent:
        print
        model = GraphRecurrent(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.dt:
        # model = GNNBlock(dataset.inp_dim, dataset.out_dim)
        model = GNNDTNet(width= 128, in_channels=1)
    else:
        sys.exit()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=FLAGS.lr)#, weight_decay=2e-4)
    return model, optimizer


def test(train_dataloader, model, FLAGS, step=0, gen=False):
    global best_test_error, best_gen_test_error
    if FLAGS.cuda:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    replay_buffer = None
    dist_list = []
    energy_list = []
    min_dist_energy_list = []

    model.eval()
    counter = 0
    with torch.no_grad():
        for data in train_dataloader:
            data = data.to(dev)
            # print(data.x.shape)
            # print(data.y.shape)
            im = data['y']

            pred = (torch.rand_like(data.edge_attr) - 0.5) * 2.

            scratch = torch.zeros_like(pred)

            pred_init = pred
            pred, preds, im_grad, energies, scratch = gen_answer(data, FLAGS, model, pred, scratch, FLAGS.num_steps, create_graph=False)

            preds = torch.stack(preds, dim=0)
            energies = torch.stack(energies, dim=0).mean(dim=-1).mean(dim=-1)
            # print("energies shape is ", energies.shape)

            dist = (preds - im[None, :])
            dist = torch.pow(dist, 2).mean(dim=-1)
            n = dist.size(1)

            dist = dist.mean(dim=-1)
            min_idx = energies.argmin(dim=0)
            # print("min index is ", min_idx)
            # print("dist shape is ", dist.shape)
            # print(dist)
    
            dist_min = dist[min_idx]
            min_dist_energy_list.append(dist_min)
            # print("dist min is ",dist_min)
            # sys.exit()

            dist_list.append(dist.detach())
            energy_list.append(energies.detach())

            counter = counter + 1

            if counter > 10:
                # print("dist shape is ", torch.stack(dist_list, dim=0).shape)
                # print("dist shape is ",torch.stack(dist_list, dim=0).mean(dim=0).shape)
                
                dist = torch.stack(dist_list, dim=0).mean(dim=0)
                # print(dist)
                energies = torch.stack(energy_list, dim=0).mean(dim=0)
                # print(energies)
                
                min_dist_energy = torch.min(dist)#torch.stack(min_dist_energy_list, dim=0).mean()
                # print(min_dist_energy)
                # sys.exit()
                print("Testing..................")
                print("last step error: ", dist)
                print("energy values: ", energies)
                print('test at step %d done!' % step)
                break

    if gen:
        best_gen_test_error = min(best_gen_test_error, min_dist_energy.item())
        print("best gen test error: ", best_gen_test_error)
    else:
        # print("best error is ", best_test_error)
        # print("min dist energy ", min_dist_energy.item())
        best_test_error = min(best_test_error, min_dist_energy.item())
        print("best test error: ", best_test_error)
    model.train()

def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def progressive_loss(model, data, FLAGS):
    alpha = FLAGS.alpha
    inputs = data
    targets = data.y
    max_iters = FLAGS.num_steps
    net = model
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    if alpha != 1:
        outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                    outputs_max_iters.size(1), -1)
        loss_max_iters = criterion(outputs_max_iters, targets)
    else:
        loss_max_iters = torch.zeros_like(targets).float()

    # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
    # so we save time by setting it equal to 0).
    if alpha != 0:
        outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        loss_progressive = criterion(outputs, targets)
    else:
        loss_progressive = torch.zeros_like(targets).float()

    loss_max_iters_mean = loss_max_iters.mean()
    loss_progressive_mean = loss_progressive.mean()

    loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean

    return loss
# def MSE_loss():

def train(train_dataloader, test_dataloader, gen_dataloader, logger, model, optimizer, FLAGS, logdir, rank_idx):
    it = FLAGS.resume_iter
    optimizer.zero_grad()
    num_steps = FLAGS.num_steps

    dev = torch.device("cuda")
    replay_buffer = ReplayBuffer(10000)
    multiples_of_500 = [x*500 for x in range(1, 21)]
    multiples_of_1000 = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,2500,5000], gamma=0.01)
    for epoch in range(FLAGS.num_epoch):
        for data in train_dataloader:
            pred = (torch.rand_like(data.edge_attr) - 0.5) * 2
            data['noise'] = pred
            scratch = torch.zeros_like(pred)
            nreplay = FLAGS.batch_size

            if FLAGS.replay_buffer and len(replay_buffer) >= FLAGS.batch_size:
                data_list, levels = replay_buffer.sample(32)
                new_data_list = data.to_data_list()

                ix = int(FLAGS.batch_size * 0.5)

                nreplay = len(data_list) + 40
                data_list = data_list + new_data_list
                data = Batch.from_data_list(data_list)
                pred = data['noise']
            else:
                levels = np.zeros(FLAGS.batch_size)
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        FLAGS.batch_size) > 1.0)

            data = data.to(dev)
            pred = data['noise']
            pred, preds, im_grads, energies, scratch = gen_answer(data, FLAGS, model, pred, scratch, num_steps)
            energies = torch.stack(energies, dim=0)

            # Choose energies to be consistent at the last step
            preds = torch.stack(preds, dim=1)

            im_grads = torch.stack(im_grads, dim=1)
            im = data['y']

            if FLAGS.ponder:
                im_loss = torch.pow(preds[:, :] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            else:
                im_loss = torch.pow(preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)

            loss = im_loss.mean()
            loss.backward()

            if FLAGS.replay_buffer:
                data['noise'] = preds[:, -1].detach()
                data_list = data.cpu().to_data_list()

                replay_buffer.add(data_list[:nreplay])

            if FLAGS.gpus > 1:
                average_gradients(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if it > 10000:
                assert False

            if it % FLAGS.log_interval == 0 and rank_idx == 0:
                loss = loss.item()
                kvs = {}

                kvs['im_loss'] = im_loss.mean().item()

                string = "Iteration {} ".format(it)

                for k, v in kvs.items():
                    string += "%s: %.6f  " % (k,v)
                    logger.add_scalar(k, v, it)

                print(string)
                print(scheduler.get_last_lr())

            if it % FLAGS.save_interval == 0 and rank_idx == 0 and it>0:
                model_path = osp.join(logdir, "model_latest.pth".format(it))
                ckpt = {'FLAGS': FLAGS}

                ckpt['model_state_dict'] = model.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)

                print("Testing performance .......................")
                test(test_dataloader, model, FLAGS, step=it, gen=False)

                print("Generalization performance .......................")
                test(gen_dataloader, model, FLAGS, step=it, gen=True)
            # if it > 500:
            #     num_steps = 2
            # elif it > 1000:
            #     num_steps = 3
            it += 1



def main_single(rank, FLAGS):
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank
    world_size = FLAGS.nodes * FLAGS.gpus


    if not os.path.exists('result/%s' % FLAGS.exp):
        try:
            os.makedirs('result/%s' % FLAGS.exp)
        except:
            pass

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

    if FLAGS.gen:
        test_dataset = gen_dataset

    shuffle=True
    sampler = None

    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:8113', world_size=world_size, rank=rank_idx, group_name="default")

    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    FLAGS_OLD = FLAGS

    if FLAGS.resume_iter != 0:
        model_path = osp.join(logdir, "model_latest.pth".format(FLAGS.resume_iter))

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        FLAGS = checkpoint['FLAGS']

        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.vis = FLAGS_OLD.vis
        FLAGS.ponder = FLAGS_OLD.ponder
        FLAGS.recurrent = FLAGS_OLD.recurrent
        FLAGS.no_truncate_grad = FLAGS_OLD.no_truncate_grad
        FLAGS.gen_rank = FLAGS_OLD.gen_rank
        FLAGS.step_lr = FLAGS_OLD.step_lr

        model, optimizer = init_model(FLAGS, device, dataset)
        state_dict = model.state_dict()

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model, optimizer = init_model(FLAGS, device, dataset)
    # print("getting data")
    if FLAGS.gpus > 1:
        sync_model(model)
    # print("getting data")
    train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    gen_dataloader = DataLoader(gen_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size // 2, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    if FLAGS.train:
        model.train()
    else:
        model.eval()

    if FLAGS.train:
        # print("go train")
        train(train_dataloader, test_dataloader, gen_dataloader, logger, model, optimizer, FLAGS, logdir, rank_idx)
    else:
        # print("go test")
        test(test_dataloader, model, FLAGS, step=FLAGS.resume_iter)


def main():
    FLAGS = parser.parse_args()
    FLAGS.replay_buffer = not FLAGS.no_replay_buffer
    FLAGS.vary = True
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        pass
    main()