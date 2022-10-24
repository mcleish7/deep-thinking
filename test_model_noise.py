#based on test_model and testing

import torch
import logging
from omegaconf import OmegaConf
import deepthinking as dt
import os
from tqdm import tqdm
from deepthinking.utils.mazes_data import prepare_maze_loader

def main():
    print("in main")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("test_model_noise.py main() running.")
    model_path = "batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn" #TODO add a model into path here
    model_path_1 = "batch_shells_maze/outputs/mazes_ablation/training-abased-Paden" 
    model = "dt_net_recall_2d"
    width = "128"
    # training_args = OmegaConf.load(os.join(model_path ,".hydra/config.yaml")) 
    # cfg_keys_to_load = [("hyp", "alpha"),
    #                     ("hyp", "epochs"),
    #                     ("hyp", "lr"),
    #                     ("hyp", "lr_factor"),
    #                     ("model", "max_iters"),
    #                     ("model", "model"),
    #                     ("hyp", "optimizer"),
    #                     ("hyp", "train_mode"),
    #                     ("model", "width")]
                        
    loaders = prepare_maze_loader(train_batch_size=50,
                                   test_batch_size=25,
                                   train_data=9,
                                   test_data=33)
    print("in main 1")
    model_path = os.join(model_path ,"model_best.pth")
    state_dict = torch.load(model_path, map_location=device)
    net = dt.get_model(model, width, max_iters=50)
    net = net.to(device)
    net = net.load_state_dict(state_dict["net"])
    print("in main 2")
    log.info("==> Starting testing for 50...")
    input, target, output = save_at_50(net, [loaders["test"]], device)
    print("in main 3")
    print(input)
    print(output)
    print(target)


def save_at_50(net, testloader, device):
    max_iters = 50
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0
    switch = True
    while switch == True:
        with torch.no_grad():
            for inputs, targets in tqdm(testloader, leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs, iters_to_do=max_iters)

    return inputs, targets, outputs

if __name__ == "__main__":
    main()