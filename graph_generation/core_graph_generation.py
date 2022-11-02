from pathlib import Path
import numpy as np
import math
import networkx as nx
import random
from datetime import datetime
import argparse
import os

def graph_creation(n, method = "random"): #n=number of nodes in graph
    # Create a graph randomly
    if method == "random":
        p = min((math.log(n,2)/n),0.5)
        graph = nx.fast_gnp_random_graph(n, p)
    elif method == "complete":
        graph = nx.complete_graph(n)
    elif method == "path":
        graph = nx.path_graph(n)
    try:
        end = random.randrange(n)
        sp = nx.shortest_path(graph, 0, end) #finds shortest path between node 0 and some node
        sp_len = np.array([len(sp)])
    except:
        sp_len = np.array([-1])
    return graph,sp_len

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

def create_into_file_always_sp(n,x,method="random",train = True): #if want to always have a shortest path that exists
    test_train = "train" if train else "test"
    print(test_train)
    dir = f"~/Desktop/graph_generation_files/graph_{test_train}_{n}/"
    dir = os.path.expanduser(dir)
    Path(dir).mkdir(parents=True, exist_ok=True)
    graphs = []
    sps = []
    for i in range(0,x):
        print("Beginning iteration ",i)
        switch = False
        while switch == False:
            arr, sp = graph_creation(n, method)
            #append to file
            test = np.array([-1])
            if not(np.array_equal(sp,test)):
                graphs.append(arr)
                sps.append(sp)
                switch = True

    arr_graphs = [nx.to_numpy_array(graphs[0])]
    for i in range(1,len(graphs)):
        arr_graphs = np.append(arr_graphs,[nx.to_numpy_array(graphs[i])], axis=0)

    arr_sps= sps[0]
    for i in range(1,len(sps)):
        arr_sps = np.vstack((arr_sps,sps[i]))

    file_path = os.path.join(dir,'inputs.npz')
    np.savez(file_path, arr_graphs) #save as an npzfile
    file_path = os.path.join(dir,'solutions.npz')
    np.savez(file_path, arr_sps)

def main():
    parser = argparse.ArgumentParser(prog = 'Graph generation',description="Generator parser")
    parser.add_argument("--method", type=str, default="random", help="method to make graphs by")
    parser.add_argument("--sp", type=bool, default=False, help="guarentees a shortest path exists if true")
    parser.add_argument("--n", type=int, default=5, help="number of nodes in graphs")
    parser.add_argument("--x", type=int, default=10, help="number of graphs")
    parser.add_argument("--train", type=bool, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.sp == False:
        create_into_file(args.n, args.x, args.method, args.train)
    else:
        create_into_file_always_sp(args.n, args.x, args.method, args.train)

if __name__ == "__main__":
    main()

#python3.9 core_graph_generation.py --n 5 --x 2 --sp True --train True
#costs about 5500 bits for 10,000 6 node graphs
