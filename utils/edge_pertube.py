import copy
import networkx as nx
import random

def PerturbedProcessing(G1,rand_portion, graphname):

        G3 = copy.deepcopy(G1)
        #Input 2 graphs
        G3, G4 = perturb_edge_pair(G3, rand_portion)
        G3, G4 = ReorderingSame(G3,G4) #0505 disabled
        
        #export to the files
        nx.write_edgelist(G3, "{}1_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
        nx.write_edgelist(G4, "{}2_ran{}.edges".format(graphname, rand_portion),delimiter=',',data=False)
        print('exporting data complete')
        
        return G3, G4

''' perturbation '''
def perturb_edge_pair(G, rand_portion = 0.1):
    
    G_copy = copy.deepcopy(G)
    
    edgelist = list(G.edges)
    
    num_mask_rand = int(len(edgelist)*rand_portion)
    if rand_portion == 0:
        return G, G_copy
    
    for _ in range(num_mask_rand):
        e = random.sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G.degree[start_vertex] >= 2 and G.degree[end_vertex] >= 2:
            G.remove_edges_from(e)
            
    for _ in range(num_mask_rand):
        e = random.sample(list(edgelist),1)
        
        start_vertex = e[0][0]
        end_vertex = e[0][1]
        
        if G_copy.degree[start_vertex] >= 2 and G_copy.degree[end_vertex] >= 2:
            G_copy.remove_edges_from(e)
            
    return G, G_copy

def ReorderingSame(G1,G2):
    G1 = nx.convert_node_labels_to_integers(G1, first_label=1, ordering='default', label_attribute=None)
    G2 = nx.convert_node_labels_to_integers(G2, first_label=1, ordering='default', label_attribute=None)
    return G1,G2