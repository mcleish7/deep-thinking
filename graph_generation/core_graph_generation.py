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
        sp = np.array(sp)
    except:
        sp = np.array([-1])
    return graph,sp

def create_into_file(n,x,method="random",filename = None): #n=number of nodes in each graph, x=number of graphs in file, method=way to generate
    dir = f"~/Desktop/graph_generation_files/graph_{n}/"
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
    graph_file = filename+"-graphs"
    sp_file = filename+"-sp"
    file_path = os.path.join(dir,graph_file)
    np.savez(file_path+'.npz', *graphs) #save as an npzfile
    file_path = os.path.join(dir,sp_file)
    np.savez(file_path+'.npz', *sps)

def create_into_file_always_sp(n,x,method="random",filename = None): #if want to always have a shortest path that exists
    dir = f"~/Desktop/graph_generation_files/graph_{n}/"
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
    
    if filename == None:
        now = datetime.now().strftime("%m%d-%H.%M")
        filename = f"graph_generated_by_{method}_at_{now}"
    file_path = os.path.join(dir,'graph.npz')
    np.savez(file_path, *graphs) #save as an npzfile
    file_path = os.path.join(dir,'sp.npz')
    np.savez(file_path, *sps)

def main():
    parser = argparse.ArgumentParser(description="Generator parser")
    parser.add_argument("--method", type=str, default="random", help="method to make graphs by")
    parser.add_argument("--filename", type=str, default=None, help="name of file graph matracies will be stored in")
    parser.add_argument("--sp", type=bool, default=False, help="guarentees a shortest path exists if true")
    parser.add_argument("--n", type=int, default=5, help="number of nodes in graphs")
    parser.add_argument("--x", type=int, default=10, help="number of graphs")
    args = parser.parse_args()
    if args.sp == False:
        create_into_file(args.n,args.x,method = args.method, filename = args.filename)
    else:
        create_into_file_always_sp(args.n,args.x,method = args.method, filename = args.filename)

if __name__ == "__main__":
    main()

#python3.9 core_graph_generation.py --n 5 --x 2 --sp True --filename hi
#costs about 5500 bits for 10,000 6 node graphs
