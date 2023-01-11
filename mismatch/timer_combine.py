import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

labels = ["0.0","0.01","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"]
sums512 = [0.9405463933944702,1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9987200498580933, 1.0, 1.0]
# for alpha: 10 the time array is [0.9405463933944702, 1.0]
df = pd.DataFrame([labels,sums512]).T.astype(float)
df.columns = ['Alpha', 'rank']
# print(df.dtypes)
# print(df)
rank_names = ["AA Score", "Peturbation Score", "Speed Score", "Multiplied Score", "Mean Score"]
filepaths = ["sums_peturb.json","sums_speed.json"]
dataframes = []
for json_name in filepaths:
        with open(json_name) as f:
            data = json.load(f)
            temp = pd.DataFrame.from_dict(data)[["Alpha","rank"]].astype(float)
            dataframes.append(temp)
# print("length dataframes is",len(dataframes))
# print(dataframes[0].dtypes)
# print(dataframes[1].dtypes)
# print(dataframes[0])
# print(dataframes[1])
for i in range(0,len(dataframes)):
    df = pd.merge(df,dataframes[i], on="Alpha", how="inner", sort = True, suffixes=("",str(i)))

df['Multiplied Score'] = df['rank'] * df['rank0'] * df["rank1"]
df['Mean Score'] = df[['rank', 'rank0', 'rank1']].mean(axis=1)
print(df)
df.plot.line(x = 'Alpha')
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.ylabel("Score")
plt.title("Scores over varying alphas")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), labels = rank_names)
plt.savefig("timer_combined", bbox_inches="tight", dpi=500)

    
