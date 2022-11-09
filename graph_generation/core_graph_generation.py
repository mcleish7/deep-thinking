from pathlib import Path
import numpy as np
import math
import networkx as nx
import random
from datetime import datetime
import argparse
import os
from node2vec import Node2Vec
import sys

# Taken from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def graph_creation(n, method = "random"): #n=number of nodes in graph
    # Create a graph randomly
    if method == "random":
        p = min((math.log(n,2)/n),0.5)
        graph = nx.fast_gnp_random_graph(n, p)
    elif method == "complete":
        graph = nx.complete_graph(n)
    elif method == "path":
        graph = nx.path_graph(n)
    elif method == "empty":
        graph = nx.empty_graph(n)
    store = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            try:
                # end = random.randrange(n)
                sp = nx.shortest_path(graph, i,j) #finds shortest path between node i and j
                # print(sp)
                store[i,j] = len(sp) -1
                # sp_len = np.array([len(sp)])
            except:
                # sp_len = np.array([-1])
                store[i,j] = -1
    # print("graph is ",graph)
    # print(store)
    return graph,store

def create_into_file(n,x,method="random",train = True): #n=number of nodes in each graph, x=number of graphs in file, method=way to generate
    dir = f"~/Desktop/graph_generation_files/graph_{'train' if train else 'test'}_{n}/"
    dir = os.path.expanduser(dir)
    Path(dir).mkdir(parents=True, exist_ok=True)
    graphs = []
    sps = []
    for i in range(0,x):
        arr, sp = graph_creation(n, method)
        #append to file
        print(np.asarray(arr).shape)
        graphs.append(arr)
        sps.append(sp)
    
    if filename == None:
        now = datetime.now().strftime("%m%d-%H.%M")
        filename = f"graph_generated_by_{method}_at_{now}"

    file_path = os.path.join(dir,'inputs.npz')
    np.savez(file_path, *graphs) #save as an npzfile
    file_path = os.path.join(dir,'solutions.npz')
    np.savez(file_path, *sps)

def toFile(n,train, arr_graphs, arr_sps):
    test_train = "train" if train else "test"
    dir = f"~/Desktop/graph_generation_files/graph_{test_train}_{n}/"
    dir = os.path.expanduser(dir)

    Path(dir).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(dir,'inputs.npz')
    np.savez(file_path, arr_graphs) #save as an npzfile

    file_path = os.path.join(dir,'solutions.npz')
    np.savez(file_path, arr_sps)

def embedn2v(graphs,n):
    temp = []
    # blockPrint() #node2vec puts alot of unnessacary printing in the logs so I suppress this
    for graph in graphs:
        # Precompute probabilities and generate walks
        node2vec = Node2Vec(graph, dimensions=n, walk_length=30, num_walks=200, workers=4)  #Use temp_folder for big graphs
        #dimensions is the size of the output array for each node e.g. 5 nodes and 64 dimensions gives [5,64] output
        # Embed nodes
        model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

        temp.append(model.wv.vectors)
    return temp

def create_always_sp(n,x,method="random",train = True, save = False, embed = None): #if want to always have a shortest path that exists
    graphs = []
    sps = []
    for i in range(0,x):
        print("Beginning iteration ",i)
        # switch = False
        # while switch == False:
        arr, sp = graph_creation(n, method)
            #append to file
            # test = np.array([-1])
            # if not(np.array_equal(sp,test)):
        graphs.append(arr)
        sps.append(sp)
                # switch = True

    if embed == "n2v":
        graphs = embedn2v(graphs,n)
    else: 
        for i in range(0,len(graphs)):
            graphs[i]=nx.to_numpy_array(graphs[i])
    arr_graphs = [graphs[0]]
    for i in range(1,len(graphs)):
        arr_graphs = np.append(arr_graphs,[graphs[i]], axis=0)

    arr_sps = [sps[0]]
    for i in range(1,len(sps)):
        arr_sps = np.append(arr_sps,[sps[i]], axis=0)
    if save == True:
        toFile(n,train, arr_graphs, arr_sps)
    # print("arr_graphs is ",arr_graphs)

def main():
    parser = argparse.ArgumentParser(prog = 'Graph generation',description="Generator parser")
    parser.add_argument("--method", type=str, default="random", help="method to make graphs by")
    parser.add_argument("--embed", type=str, default=None, help="method to embed graphs by")
    parser.add_argument("--sp", type=bool, default=False, help="guarentees a shortest path exists if true")
    parser.add_argument("--save", type=bool, default=False, help="saves to file if true",action=argparse.BooleanOptionalAction)
    parser.add_argument("--n", type=int, default=5, help="number of nodes in graphs")
    parser.add_argument("--x", type=int, default=10, help="number of graphs")
    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.sp == False:
        create_into_file(args.n, args.x, args.method, args.train)
    else:
        create_always_sp(args.n, args.x, args.method, args.train, args.save, args.embed)

if __name__ == "__main__":
    main()

#python3.9 core_graph_generation.py --n 5 --x 2 --sp True --train --save
#costs about 5500 bits for 10,000 6 node graphs
