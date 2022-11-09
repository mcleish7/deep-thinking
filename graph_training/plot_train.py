#get_df and get_little_df_from_one_run taken from data_anlysis
from matplotlib import pyplot as plt
import os

import argparse
import copy
import glob
import json
from tabulate import tabulate

import pandas as pd


def get_little_df_from_one_run(data_dict):
    num_new_dicts = len(data_dict["test_iters"])
    test_iters = data_dict["test_iters"]
    out_dict = {}
    for i in range(num_new_dicts):
        new_dict_i = copy.deepcopy(data_dict)
        new_dict_i["test_acc"] = data_dict["test_acc"][str(test_iters[i])] \
            if data_dict["test_acc"] else 0
        new_dict_i["val_acc"] = data_dict["val_acc"][str(test_iters[i])] \
            if data_dict["val_acc"] else 0
        new_dict_i["train_acc"] = data_dict["train_acc"][str(test_iters[i])] \
            if data_dict["train_acc"] else 0
        new_dict_i["test_iter"] = test_iters[i]
        out_dict[i] = new_dict_i
    little_df = pd.DataFrame.from_dict(out_dict, orient="index")
    return little_df


def get_df(filepath, acc_filter=None):
    pd.set_option("display.max_rows", None)
    df = pd.DataFrame()
    num_checkpoints = 0
    for f_name in glob.iglob(f"{filepath}/**/*testing*/stats.json", recursive=True):
        num_checkpoints += 1
        with open(f_name, "r") as fp:
            data = json.load(fp)
        if acc_filter is not None:
            m = data["max_iters"]
            if data["train_acc"][str(m)] > acc_filter:
                df = df.append(get_little_df_from_one_run(data))
        else:
            df = df.append(get_little_df_from_one_run(data))
    num_trained = len(df)
    return df, num_checkpoints, num_trained

def graph_time(arr):
    plt.clf()
    plt.plot(arr)
    plt.title('Iterations to recover')
    save_path = os.path.join("test_time","test_time_correctness.png")
    plt.savefig(save_path)

names= []
dfs = []
for name in names:
    filePath = f"outputs/graphs_abalation/training-{name}"
    df, num_checkpoints, num_trained = get_df(filePath)
    dfs.append(df)
    df.plot()


