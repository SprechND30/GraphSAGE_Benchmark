import numpy as np
from utils import load_data as load
import tensorflow as tf
import json
import networkx as nx
from networkx.readwrite import json_graph as jg

flags = tf.app.flags
FLAGS = flags.FLAGS

# Dataset and destination directory
flags.DEFINE_string('dataset', None, 'Dataset to be used (citeseer/cora).')
flags.DEFINE_string('destination_dir', None, 'Directory to which the data files will be sent.')


##
# Returns a graph, constructed out of the inputted adjacency, label, and feature matrices, 
# as well as the test and validation masks.
##
def create_G_idM_classM(adjacency, features, testMask, valMask, labels):
    
    # 1. Create Graph
    print("Creating graph...")
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency)
    num_nodes = G.number_of_nodes()
    
    # Change labels to int from numpy.int64
    labels = labels.tolist()
    for arr in labels:
        for integer in arr:
            integer = int(integer)
    
    # Iterate through each node, adding the features
    i = 0
    for n in list(G):
        G.nodes[i]['features'] = list(map(float, list(features[i])))
        G.nodes[i]['test'] = bool(testMask[i])
        G.nodes[i]['val'] = bool(valMask[i])
        G.nodes[i]['labels'] = list(map(int, list(labels[i])))
        i += 1
       
    # 2. Create id-Map and class-Map
    print("Creating id-Map and class-Map...")
    # Initialize the dictionarys
    idM = {}
    classM = {}
    
    # Populate the dictionarys
    i = 0
    while i < num_nodes:
        idStr = str(i)
        idM[idStr] = i
        classM[idStr] = list(labels[i])
        i += 1
    
    return G, idM, classM
    

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
    adj, features, labels, valMask, testMask = load(FLAGS.dataset)
    
    # Turn CSR matricies into numpy arrays
    adj = adj.toarray()
    features = features.toarray()
    
    # Create Graph, IDMap, and classMap
    G, IDMap, classMap = create_G_idM_classM(adj, features, testMask, valMask, labels)
    
    # Dump everything into .json files and one .npy
    #dumpJSON(FLAGS.destination_dir, FLAGS.dataset, G, IDMap, classMap, features)
    
    
if __name__ == "__main__":
    main()
