import json
import os
from omegaconf import OmegaConf
import glob
from tabulate import tabulate
import pandas as pd
import dataframe_image as dfi
import sys
import numpy as np

name = "time"
# filepaths = ["outputs/test_maze_59"]
filepaths = ["outputs/combined_for_timer"]
count = 0
alphas = []
maxes = []
for filepath in filepaths:
    for f_name in glob.iglob(f"{filepath}/**/*testing*/stats.json", recursive=True):
        model_path = "/".join(f_name.split("/")[1:-1])
        json_name = os.path.join("/".join(f_name.split("/")[:-1]), "stats.json")
        with open(json_name) as f:
            data = json.load(f)
            arr = np.array(list(data["test_acc"].values()))
            arr = np.delete(arr, 0)
        max = np.argmax(arr)

        if not (arr[max] >= 99.0):
            max = arr.shape[0]
        alphas.append(round(data["alpha"],2))
        maxes.append(max)
        count+=1

maxs = np.array(maxes)

normed = abs(1-(maxs-np.min(maxs))/(np.max(maxs)-np.min(maxs)))
checkpoints = []
for i in range(0,len(alphas)):
    checkpoints.append([
        alphas[i],maxs[i],normed[i]
    ])
head = ["Alpha","Time to max", "rank"]
print(tabulate(checkpoints, headers=head))
print(f"There are {count} rows in the table")
df = pd.DataFrame(checkpoints, columns = head)
df.to_json("sums_speed.json")