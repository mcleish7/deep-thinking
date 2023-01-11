import torch
import numpy as np
import deepthinking.models as models
import deepthinking as dt
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
import os
from tqdm import tqdm
import nvidia_smi
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

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
    # net.eval()
    return net

def compute_cross_pi(net, testloader, iters, problem, device,alpha):
    # net.eval()
    corrects = 0
    total = 0

    idx = 0
    path_indep_val = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # print("Total memory:", info.total)
    # print("Free memory before loop:", info.free)
 
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device), targets.to(device)
            init_outputs, fp_val1 = net(inputs, return_fp=True)

            tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1, 1))
            tiled_targets = torch.tile(targets, (targets.shape[0], 1, 1))
            # print("Free memory before interleave:", info.free)
            # print("inputs shape 0 is ",inputs.shape[0])
            repeated_fp = torch.repeat_interleave(fp_val1, repeats=inputs.shape[0], dim=0)
            # nvidia_smi.nvmlShutdown()  
            next_outputs, fp_val2 = net(tiled_inputs, interim_thought=repeated_fp, return_fp=True)
            # print("Free memory after interleave:", info.free)
            total += fp_val2.size(0)

            idx = np.arange(0, tiled_inputs.shape[0], inputs.shape[0])
            fp1 = repeated_fp.view(repeated_fp.shape[0], -1)
            fp2 = fp_val2.view(fp_val2.shape[0], -1)
            
            bsz = inputs.shape[0]
            for i in range(inputs.shape[0]):
                cur_idx = idx + i
                conseq_idx = np.arange(i*bsz, i*bsz + inputs.shape[0])
                path_indep_val += cos(fp1[cur_idx], fp2[conseq_idx]).sum()
            # break
        print("for net ",alpha," Cosine similarity", path_indep_val/total)
    return path_indep_val/total

# class Problems:
#   name = "mazes"

# problem = Problems()

@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    problem = cfg.problem

    # print("train batch size is ",problem.hyp.train_batch_size)
    # print("test batch size is ",problem.hyp.test_batch_size)
    loaders = dt.utils.get_dataloaders(problem)
    cwd = os.getcwd()
    # print("cwd is ",cwd)
    os.chdir('../../..')
    cwd = os.getcwd()
    # print("cwd is ",cwd)
    testloader = loaders["test"]#[loaders["test"], loaders["val"], loaders["train"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iters = 300
    aa = []
    alphas = [-1,1,2,3,4,5,6,7,8,9]
    for alpha in alphas:
        net = get_net(device, alpha)
        aa.append(compute_cross_pi(net, testloader, iters, problem, device, alpha).tolist())
        file_name = f"score_{problem.name}_{problem.train_data}.txt"
        with open(file_name, 'w+') as f:
            f.write(f"for alpha: {alpha} the time array is {aa}")

# if __name__ == "__main__":

#     run_id = dt.utils.generate_run_id()
#     sys.argv.append(f"+run_id={run_id}")
#     main()

# python3.9 score_email.py problem=prefix_sums problem.hyp.test_batch_size=10 problem.hyp.train_batch_size=10 problem.test_data=512 problem.train_data=32
# maze models:
# for alpha: 8 the time array is [0.9971649646759033, 0.9980168342590332, 0.9968770146369934, 0.931258499622345, 0.9408687949180603, 0.7856437563896179, 0.9429271817207336, 0.9065360426902771, 0.8678450584411621]
# for alpha: 9 the time array is [0.916424572467804]

m1 = [0.9971649646759033, 0.9980168342590332, 0.9968770146369934, 0.931258499622345, 0.9408687949180603, 0.7856437563896179, 0.9429271817207336, 0.9065360426902771, 0.8678450584411621]
m2 = [0.916424572467804]
maze9 = m1+m2
# for alpha: 9 the time array is 
maze9 = [0.9981143474578857, 0.9999440312385559, 0.9997977018356323, 0.992916464805603, 0.9892067909240723, 0.9873379468917847, 0.9823738932609558, 0.9916004538536072, 0.9860827922821045, 0.9915866851806641]
maze59 = [0.9758304953575134, 0.9607346057891846, 0.963686466217041, 0.9220239520072937, 0.8237228989601135, 0.6947765946388245, 0.8579129576683044, 0.8902077078819275, 0.8086482882499695, 0.9082356691360474]
# for alpha: 9 the time array is 
maze59 = [0.9758304953575134, 0.9607346057891846, 0.963686466217041, 0.9220239520072937, 0.8237228989601135, 0.9195083379745483, 0.8579129576683044, 0.8902077078819275, 0.8086482882499695, 0.9082356691360474]
# for alpha: 9 the time array is 
maze13 = [0.9974822402000427, 0.9999359846115112, 0.9995937943458557, 0.9902072548866272, 0.9831128120422363, 0.985153317451477, 0.9757426381111145, 0.978689968585968, 0.9815245270729065, 0.9803295731544495]

sums32 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9987200498580933, 1.0]
sums512 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9987200498580933, 1.0]
#changed nan to 0.0
chess7 = [0.0, 0.005817557219415903, 0.8072536587715149, 0.9524214863777161, 0.9572643041610718, 0.9805848598480225, 0.9854093194007874, 0.9914458394050598, 0.9960649609565735, 0.9952036142349243]
chess11 = [0.0, 0.006417518015950918, 0.8050198554992676, 0.9460858702659607, 0.9505954384803772, 0.9736934900283813, 0.9806694984436035, 0.9879935383796692, 0.9943339228630066, 0.9931629300117493]

def graph_array(maze9, maze59, sums32, sums512, chess7, chess11):
  f, ax = plt.subplots(1, 1)
  ax.plot(maze9, label = "Maze 13x13")
  ax.plot(maze59, label = "Maze 59x59")
  ax.plot(sums32, label = "Sums 32bits")
  ax.plot(sums512, label = "Sums 512bits")
  ax.plot(chess7, label = "Chess [600k-700k]")
  ax.plot(chess11, label = "Chess [1M-1.1M]")
  labels = ["0.01","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
  ticks = [0,1,2,3,4,5,6,7,8,9]
  plt.xticks(ticks=ticks, labels=labels)
  ax.set(xlabel='Alpha Value', ylabel='AA score', title="AA score for maze models over alpha values")
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#   plt.grid()
  plt.savefig("mazes_AA_score", bbox_inches="tight", dpi=500)

graph_array(maze13, maze59, sums32, sums512, chess7, chess11)