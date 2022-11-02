#code checks if file has been created if not creates it
# from pathlib import Path

# Path("graph_generation_files").mkdir(parents=True, exist_ok=True)
# Path("testing_1").mkdir(parents=True, exist_ok=True)

import sys
import numpy as np
# container = np.load('graph_generated_by_random_at_1102-10.03.npz')
# # print(data.size())
# # print(type(container))
# # data = [container[key] for key in container]
# # print(type(data))
# # print(len(data))
# # for i in data:
# #     print(data[i])
# #     sys.exit()

# for item in container.keys():
#     print(item)
#     print(container[item])
#     print(type(container[item]))
#     sys.exit()

import networkx as nx
graph = nx.complete_graph(5)
arr = nx.to_numpy_array(graph)
print(arr)
print(type(arr))
sp = nx.shortest_path(graph, 0, 4)
sp =np.array(sp)
print(sp)
print(type(sp))