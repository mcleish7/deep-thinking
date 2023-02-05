import ast
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sys

name = "dt"
file = f"loss_json/loss_{name}.txt"
with open(file, 'r') as f:
    x = f.readlines()
    data = ast.literal_eval(x[1])

data_200 = data[200:]
data = data[5:]
data = data[200:]
plt.axhline(y=min(data_200), color='r')
plt.axhline(y=max(data_200), color='r')
plt.plot(data, linewidth=0.5)
extraticks = [min(data_200)]
if max(data_200) <=2.5:
    extraticks.append(max(data_200))
# plt.ylim([0, 2.5])
plt.yticks(list(plt.yticks()[0])+extraticks)
filename = f"loss_{name}"
plt.savefig(filename)