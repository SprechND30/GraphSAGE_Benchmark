import numpy as np
import tensorflow as tf
import json
from networkx.readwrite import json_graph as jg
import os

flags = tf.app.flags
FLAGS = flags.FLAGS

# Dataset and destination directory
flags.DEFINE_string('dataset', None, 'Dataset to be used (reddit/reddit).')
flags.DEFINE_string('destination_dir', None, 'Directory to which the data files will be sent.')
flags.DEFINE_float('pollute_ratio', 0.2, 'ratio of nodes to pollute.')
flags.DEFINE_float('attribute_pollution_ratio', 0.2, 'ratio of nodes to pollute.')
    

##
# Load data with specified prefix from its constituent files. Adapted from GraphSAGE's
# utils.py method 'load_data().' 
# Returns graph, ID map, class map, and feature vectors.
##
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


##
# Pollutes graph, randomly selecting features from randomly selected nodes. The pollute ratio and
# attribute pollution ratio flags determine the probability that a given node or attribute will
# be corrupted, respectively. Also labels node accordingly, in graph and class map.
# Returns graph, class map, and feature vectors.
##
def pollute_graph(G, idMap, classMap, feats):
    print ("Polluting data\n")
    
    # Number of nodes, number of nodes in validation and test sets
    num_nodes = G.number_of_nodes()
    num_val = 500
    num_test = 1000
    
    # Number of polluted nodes in train, validationm, and test sets, respectively
    poll_num_train = int((num_nodes - (num_val+num_test)) * FLAGS.pollute_ratio)
    poll_num_val = int(num_val * FLAGS.pollute_ratio)
    poll_num_test = int(num_test * FLAGS.pollute_ratio)
    
    # Index of first validation node and first test node, respectively
    idx_val = (num_nodes - 1) - (num_val + num_test)
    idx_test = (num_nodes - 1) - (num_test)
    
    # Arrays of the indices of polluted nodes in each of the three sets
    poll_idx_train = np.random.choice(idx_val-1, poll_num_train, replace=False)
    poll_idx_val = np.random.choice(range(idx_val, idx_test-1), poll_num_val, replace=False)
    poll_idx_test = np.random.choice(range(idx_test, num_nodes-1), poll_num_test, replace=False)
    
    # The number of attributes and polluted attributes in the feature vector of a node
    attr_dim = len(G.nodes[0]['feature'])
    poll_num_attr = int(attr_dim * FLAGS.attribute_pollution_ratio)
    
    # Iterate through each node in the graph
    for n in list(G):
        
        # Assign to train, val, or test set
        if n < idx_val:
            G.nodes[n]['val'] = False
            G.nodes[n]['test'] = False
        elif idx_val <= n < idx_test:
            G.nodes[n]['val'] = True
            G.nodes[n]['test'] = False
        elif idx_test <= n:
            G.nodes[n]['val'] = False
            G.nodes[n]['test'] = True
        
        # If the node is to be polluted, proceed to its features
        if (n in poll_idx_train) or (n in poll_idx_val) or (n in poll_idx_test):
            G.nodes[n]['label'] = [0, 1]
            
            poll_attr = np.random.choice(attr_dim, poll_num_attr, replace=False)
            
            # Iterate through each of the node's features
            i = 0
            while i < poll_num_attr:
                
                # If this feature is polluted, switch its value
                if G.nodes[n]['feature'][poll_attr[i]] == 1.0:
                    G.nodes[n]['feature'][poll_attr[i]] = 0.0
                    feats[n][poll_attr[i]] = 0.0
                else:
                    G.nodes[n]['feature'][poll_attr[i]] = 1.0
                    feats[n][poll_attr[i]] = 1.0
                
                i += 1
            
            classMap[str(n)] = [0, 1]
        
        # Else, label it as clean
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
