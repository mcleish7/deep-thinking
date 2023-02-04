import json
import os
from omegaconf import OmegaConf
import glob
from tabulate import tabulate
import pandas as pd
import dataframe_image as dfi

filepaths = ["outputs/AvonTests/","../batch_reproduce_2/outputs/testing_default","../batch_reproduce_2.2/outputs/testing_default","../batch_reproduce_2.3/outputs/testing_default"]
count =0
checkpoints = []
for filepath in filepaths:
    for f_name in glob.iglob(f"{filepath}/**/*testing*/stats.json", recursive=True):
        model_path = "/".join(f_name.split("/")[1:-1])
        json_name = os.path.join("/".join(f_name.split("/")[:-1]), "stats.json")
        with open(json_name) as f:
            data = json.load(f)
        if data["model"] == "dt_net_recall_2d_width=128":
            if data["alpha"] == 1.0:
                if data["test_data"] == 59:
                    checkpoints.append([model_path,
                    round(max(data["train_acc"].values()),3),
                    round(max(data["val_acc"].values()),3),
                    round(max(data["test_acc"].values()),3)
                    ])
                    count+=1

head = ["Model Name","Train Acc","Val Acc","Test Acc"]
print(tabulate(checkpoints, headers=head))
print(f"There are {count} rows in the table")
df = pd.DataFrame(checkpoints, columns = head)
pd.set_option('display.max_colwidth', None)
# df = df.style.set_properties(**{'text-align': 'left'}) 
print(df.to_latex())
# print(df)
# dfi.export(df, 'dataframe.png')