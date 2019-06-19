import numpy as np
import tensorflow as tf
import json
import networkx as nx
from networkx.readwrite import json_graph as jg
import os
import random

flags = tf.app.flags
FLAGS = flags.FLAGS

# Dataset and destination directory
flags.DEFINE_string('dataset', None, 'Dataset to be used (reddit/reddit).')
flags.DEFINE_string('destination_dir', None, 'Directory to which the data files will be sent.')
flags.DEFINE_float('pollute_ratio', 0.2, 'ratio of nodes to pollute.')
flags.DEFINE_float('attribute_pollution_ratio', 0.2, 'ratio of nodes to pollute.')
    

def load_data(prefix, normalize=True):
    G_data = json.load(open(prefix + "-G.json"))
    G = jg.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    return G, id_map, class_map, feats


def pollute_graph(G, idMap, classMap, feats):
    print ("Polluting data\n")
    
    #num_nodes = G.number_of_nodes()
    #num_poll_node = int(num_nodes * FLAGS.pollute_ratio)
    #num_poll_attr = int(len(G.nodes[0]['feature']) * FLAGS.attribute_pollution_ratio)
    
    for n in list(G):
        isPollute = False
        rand = random.uniform(0, 1)
        
        # Is this node polluted?
        if rand < FLAGS.pollute_ratio:
            isPollute = True
        
        # Set accordingly
        if isPollute:
            G.nodes[n]['label'] = [0, 1]
            
            for f in G.nodes[n]['feature']:
                attrIsPollute = False
                randAttr = random.uniform(0, 1)
                i = int(f)
        
                # Is this node polluted?
                if randAttr < FLAGS.attribute_pollution_ratio:
                    attrIsPollute = True
                    
                if attrIsPollute:
                    if G.nodes[n]['feature'][i] == 1.0:
                        G.nodes[n]['feature'][i] = 0.0
                        feats[n][i] = 0.0
                    else:
                        G.nodes[n]['feature'][i] = 1.0
                        feats[n][i] = 1.0
            
            classMap[str(n)] = [0, 1]
        else:
            G.nodes[n]['label'] = [1, 0]
            classMap[str(n)] = [1, 0]
        
    return G, classMap, feats

##
# Dumps the inputted graph, id-map, class-map, and feature matrix into their respective
# .json files and .numpy file.
##
def dumpJSON(destDirect, datasetName, graph, idMap, classMap, features):
    
    print("Dumping into JSON files...")
    # Turn graph into data
    dataG = jg.node_link_data(graph)
    
    #Make names
    json_G_name = destDirect + '/' + datasetName + '-G.json'
    json_ID_name = destDirect + '/' + datasetName + '-id_map.json'
    json_C_name = destDirect + '/' + datasetName + '-class_map.json'
    npy_F_name = destDirect + '/' + datasetName + '-feats'
    
    # Dump graph into json file
    with open(json_G_name, 'w') as outputFile:
        json.dump(dataG, outputFile)
        
    # Dump idMap into json file
    with open(json_ID_name, 'w') as outputFile:
        json.dump(idMap, outputFile)
        
    # Dump classMap into json file
    with open(json_C_name, 'w') as outputFile:
        json.dump(classMap, outputFile)
        
    # Save features as .npy file
    print("Saving features as numpy file...")
    np.save(npy_F_name, features)


def main():
    
    # Load data
    G, idMap, classMap, feats = load_data(FLAGS.dataset)
    
    # Pollute graphs
    G, classMap, feats = pollute_graph(G, idMap, classMap, feats)
    
    datasetName = FLAGS.dataset.split("/")[-1]
    
    # Dump everything into .json files and one .npy
    dumpJSON(FLAGS.destination_dir, datasetName, G, idMap, classMap, feats)
    
    
if __name__ == "__main__":
    main()
