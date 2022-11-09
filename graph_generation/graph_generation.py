#code based on https://github.com/eliorc/node2vec, the node2vec documentation

import networkx as nx
from node2vec import Node2Vec
from pathlib import Path
import numpy as np
from datetime import datetime
import argparse
import math

def node2vec(graph,EMBEDDING_FILENAME,EMBEDDING_MODEL_FILENAME):
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  #Use temp_folder for big graphs
    #dimensions is the size of the output array for each node e.g. 5 nodes and 64 dimensions gives [5,64] output

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings
    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)
    # Save model for later use
    model.save(EMBEDDING_MODEL_FILENAME)
    return model

def to_numpy(model):
    #puts matrix in to an r x c numpy array
    arr = []
    rows = 0
    for node in model.wv.index_to_key:
        arr = np.concatenate((arr,model.wv[node]), axis=0)
        rows += 1
    columns = int(arr.size/rows)
    arr = np.reshape(arr,(rows,columns))
    return arr

def edge2vec(model,EDGES_EMBEDDING_FILENAME):
    # Embed edges using Hadamard method
    from node2vec.edges import HadamardEmbedder
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

    # Look for embeddings on the fly - here we pass normal tuples
    edges_embs[('1', '2')]

    # Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
    edges_kv = edges_embs.as_keyed_vectors()

    # Look for most similar edges - this time tuples must be sorted and as str
    edges_kv.most_similar(str(('1', '2')))

    # Save embeddings for later use
    edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)


def graph_creation(n, method = "random"): #n=number of nodes in graph
    dir = f"graph_generation_files/graph_{n}/"
    Path(dir).mkdir(parents=True, exist_ok=True)
    EMBEDDING_FILENAME = dir+"test_filename.emb"
    EMBEDDING_MODEL_FILENAME = dir+"test_model_filename.model"
    EDGES_EMBEDDING_FILENAME = dir+"test_edges_filename.edges"

    # Create a graph randomly
    if method == "random":
        p = min((math.log(n,2)/n),0.5)
        graph = nx.fast_gnp_random_graph(n, p)
    elif method == "complete":
        graph = nx.complete_graph(n)
    elif method == "path":
        graph = nx.path_graph(n)
    try:
        sp = nx.shortest_path(graph, 0, 4)
        
        # print(sp)
    except:
        sp = None
        # print(sp)
    
    model = node2vec(graph,EMBEDDING_FILENAME,EMBEDDING_MODEL_FILENAME) #runs node 2 vec

    arr = to_numpy(model) #puts into numpy array

    #TODO check if need edges part
    edge2vec(model,EDGES_EMBEDDING_FILENAME) #runs node2vec on edges

    return arr, sp

def create_into_file(n,x,method="random",filename = None): #n=number of nodes in each graph, x=number of graphs in file, method=way to generate
    graphs = []
    for i in range(0,x):
        arr, sp = graph_creation(n, method)
        #append to file
        print(np.asarray(arr).shape)
        graphs.append(arr)
    
    if filename == None:
        now = datetime.now().strftime("%m%d-%H.%M")
        filename = f"graph_generated_by_{method}_at_{now}"

    np.savez(filename+'.npz', *graphs) #save as an npzfile

def create_into_file_always_sp(n,x,method="random",filename = None): #if want to always have a shortest path that exists
    graphs = []
    for i in range(0,x):
        switch = False
        while switch == False:
            arr, sp = graph_creation(n, method)
            #append to file
            if sp != None:
                graphs.append(arr)
                switch = True
    
    if filename == None:
        now = datetime.now().strftime("%m%d-%H.%M")
        filename = f"graph_generated_by_{method}_at_{now}"

    np.savez(filename+'.npz', *graphs) #save as an npzfile

def load_npz_and_print(filename): 
    data = np.load(filename+'.npz')
    for item in data.keys():
        print(data[item]) #data[item] is a numpy array

def main():
    parser = argparse.ArgumentParser(description="Generator parser")
    parser.add_argument("--method", type=str, default="random", help="method to make graphs by")
    parser.add_argument("--filename", type=str, default=None, help="name of file graph matracies will be stored in")
    parser.add_argument("--sp", type=bool, default=False, help="guarentees a shortest path exists if true")
    args = parser.parse_args()
    if args.sp == False:
        create_into_file(5,10,method = args.method, filename = args.filename)
    else:
        create_into_file_always_sp(5,10,method = args.method, filename = args.filename)

if __name__ == "__main__":
    main()

#python3.9 graph_generation.py