from functools import reduce
import numpy as np
import networkx as nx

def create_idx_dict_pair(G1,G2):
    '''
    Make sure that this function is followed after preprocessing dict.

    '''
    
    G1list = [int(node) for node in G1.nodes()]
    #G1list.sort()
    idx1_list = list(range(G1.number_of_nodes()))
    #make dict for G1
    idx1_dict = {a : b for b, a in zip(idx1_list,G1list)}

    
    G2list = list(int(node) for node in G2.nodes())
    #G2list.sort()
    idx2_list = list(range(G2.number_of_nodes()))
    #make dict for G2
    idx2_dict = {c : d for d, c in zip(idx2_list,G2list)}
    
    return idx1_dict, idx2_dict

def get_reversed(alignment_dict):
    reversed_dictionary = {value : key for (key, value) in alignment_dict.items()}
    return alignment_dict, reversed_dictionary

def preprocessing(G1, G2, alignment_dict):
    '''
    Parameters
    ----------
    G1 : source graph
    G2 : target graph
    alignment_dict : grth dict

    '''
    # shift index for constructing union
    # construct shifted dict
    shift = G1.number_of_nodes()
    G1_list = list(G1.nodes())
    G1_int_list = [int(node) for node in G1.nodes()]
    G2_list = list(G2.nodes())
    G2_shiftlist = list(int(idx) + shift for idx in list(G2.nodes()))
    shifted_G1_dict = dict(zip(G1_list,G1_int_list))
    shifted_dict = dict(zip(G2_list,G2_shiftlist))
    
    #relable idx for G2
    G1 = nx.relabel_nodes(G1, shifted_G1_dict)
    G2 = nx.relabel_nodes(G2, shifted_dict)
    
    #update alignment dict
    align1list = list(alignment_dict.keys())
    align2list = list(alignment_dict.values())   
    shifted_align2list = [a+shift for a in align2list]
    
    groundtruth_dict = dict(zip(align1list, shifted_align2list))
    groundtruth_dict, groundtruth_dict_reversed = get_reversed(groundtruth_dict)
    
    return G1,G2, groundtruth_dict, groundtruth_dict_reversed

def aug_trimming(aug_s, aug_t):
    #concat two matrice
    concat_aug = np.concatenate((aug_s, aug_t), axis = 0)
    concat_aug = concat_aug[:,~np.all(concat_aug == 0, axis = 0)]
    aug_s_trimmed = concat_aug[:len(aug_s)]
    aug_t_trimmed = concat_aug[len(aug_s):]
    return aug_s_trimmed, aug_t_trimmed

def augment_attributes(Gs, Gt, attr_s, attr_t, num_attr, version = "katz", khop = 1, penalty = 0.1, normalize = True): 
    Gs_nodes = list(Gs.nodes())
    Gt_nodes = list(Gt.nodes())
    print(f"This is {version} binning version ")
    if version == "katz":   
        attdict_s = nx.katz_centrality_numpy(Gs,
                                              alpha = 0.01,
                                              beta = 1, 
                                              normalized = False)
        attdict_t = nx.katz_centrality_numpy(Gt,
                                              alpha = 0.01,
                                              beta = 1,
                                              normalized = False)
        
    elif version == "eigenvector":

        attdict_s = nx.eigenvector_centrality(Gs, max_iter = 500, tol = 1e-8)
        attdict_t = nx.eigenvector_centrality(Gt, max_iter = 500, tol = 1e-8)
    
    elif version == "pagerank":
        attdict_s = nx.pagerank(Gs,
                                alpha = 0.85,
                                max_iter = 100)
        attdict_t = nx.pagerank(Gt,
                                alpha = 0.85,
                                max_iter = 100)
    
    elif version == "betweenness":
        attdict_s = nx.betweenness_centrality(Gs)
        attdict_t = nx.betweenness_centrality(Gt)
    
    elif version == "closeness":
        attdict_s = nx.closeness_centrality(Gs)
        attdict_t = nx.closeness_centrality(Gt)

        
    elif version == "khop":
        
        attdict_s = {key : len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=1))                     
                     for key in Gs_nodes}
        attdict_t = {key : len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=1))
                     for key in Gt_nodes}

        attdict_s_2hop = {key : penalty * len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=2))
                     for key in Gs_nodes}        
        attdict_t_2hop = {key : penalty * len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=2))
                     for key in Gt_nodes}
        dict_seq_2_s = [attdict_s, attdict_s_2hop]
        dict_seq_2_t = [attdict_t, attdict_t_2hop]        
        attdict_s_2hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_2_s)
        attdict_t_2hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_2_t)
                
        attdict_s_3hop = {key : penalty**2 * len(nx.single_source_shortest_path_length
                            (Gs, source = key, cutoff=3))
                  for key in Gs_nodes}
        attdict_t_3hop = {key : penalty**2 * len(nx.single_source_shortest_path_length
                               (Gt, source = key, cutoff=3))
                     for key in Gt_nodes}
        dict_seq_3_s = [attdict_s_2hop, attdict_s_3hop]
        dict_seq_3_t = [attdict_t_2hop, attdict_t_3hop]        
        attdict_s_3hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_3_s)
        attdict_t_3hop = reduce(lambda d1,d2: {k: d1.get(k,0)+d2.get(k,0) for k in set(d1)|set(d2)}, dict_seq_3_t)
        
        if khop == 2:
            attdict_s = attdict_s_2hop
            attdict_t = attdict_t_2hop
            
        elif khop == 3:
            attdict_s = attdict_s_3hop
            attdict_t = attdict_t_3hop
        
        elif khop >= 4:
            print("khop should be set in range [1,3]")
            

    # Consistent normalize
    if normalize == True:        
        attdict_s = {key: (value - min(attdict_s.values())) / (max(attdict_s.values())-min(attdict_s.values()))
                     for key, value in attdict_s.items()}
        attdict_t = {key: (value - min(attdict_t.values())) / (max(attdict_t.values())-min(attdict_t.values()))
                     for key, value in attdict_t.items()}        
        interval = 1 / num_attr
        
    elif normalize == False:
        interval = max(max(attdict_s.values()),max(attdict_t.values())) / num_attr

    init_np_s = np.zeros((Gs.number_of_nodes(), num_attr))
    init_np_t = np.zeros((Gt.number_of_nodes(), num_attr))

    for idx_s, node_s in enumerate(Gs.nodes()):        # assign binning
        cent_node = attdict_s[node_s]
        init_np_s[idx_s, int(cent_node / interval) - 1] = 1

    for idx_t, node_t in enumerate(Gt.nodes()):
        cent_node = attdict_t[node_t]
        init_np_t[idx_t, int(cent_node / interval) - 1] = 1

    new_attr_s = np.append(attr_s, init_np_s, axis = 1)
    new_attr_t = np.append(attr_t, init_np_t, axis = 1)

    if len(attr_s[0]) == 1:
        new_attr_s = new_attr_s[:, 1:]
        new_attr_t = new_attr_t[:, 1:]
        
    return new_attr_s, new_attr_t