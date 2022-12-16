import torch
import numpy as np
import deepthinking.models as models
import deepthinking as dt
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
import os
from tqdm import tqdm

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

def compute_cross_pi(net, testloader, iters, problem, device):
    # net.eval()
    corrects = 0
    total = 0

    idx = 0
    path_indep_val = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device), targets.to(device)
            init_outputs, fp_val1 = net(inputs, return_fp=True)

            tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1, 1))
            tiled_targets = torch.tile(targets, (targets.shape[0], 1, 1))

            repeated_fp = torch.repeat_interleave(fp_val1, repeats=inputs.shape[0], dim=0)

            next_outputs, fp_val2 = net(tiled_inputs, interim_thought=repeated_fp, return_fp=True)
            total += fp_val2.size(0)

            idx = np.arange(0, tiled_inputs.shape[0], inputs.shape[0])
            fp1 = repeated_fp.view(repeated_fp.shape[0], -1)
            fp2 = fp_val2.view(fp_val2.shape[0], -1)
            
            bsz = inputs.shape[0]
            for i in range(inputs.shape[0]):
                cur_idx = idx + i
                conseq_idx = np.arange(i*bsz, i*bsz + inputs.shape[0])
                path_indep_val += cos(fp1[cur_idx], fp2[conseq_idx]).sum()
            
            print("Cosine similarity", path_indep_val/total)
    return path_indep_val/total

# class Problems:
#   name = "mazes"

# problem = Problems()

@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    problem = cfg.problem
    loaders = dt.utils.get_dataloaders(problem)
    cwd = os.getcwd()
    print("cwd is ",cwd)
    os.chdir('../../..')
    cwd = os.getcwd()
    print("cwd is ",cwd)
    testloader = loaders["test"]#[loaders["test"], loaders["val"], loaders["train"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = get_net(device, -1)
    iters = 300
    compute_cross_pi(net, testloader, iters, problem, device)

if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    problem = "mazes"
    name = "score_email"
    test_batch_size = 100
    train_batch_size = 100
    sys.argv.append(f"+run_id={run_id}")
    sys.argv.append(f"problem={problem}")
    sys.argv.append(f"problem.hyp.test_batch_size={test_batch_size}")
    sys.argv.append(f"problem.hyp.train_batch_size={train_batch_size}")
    main()
