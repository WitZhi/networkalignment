import json
import os
import argparse
from scipy.io import loadmat
import numpy as np
from networkx.readwrite import json_graph
from input.data_preprocess import DataPreprocess
import utils.graph_utils as graph_utils
import pickle
import torch
from sklearn.preprocessing import LabelEncoder

class Dataset:
    """
    this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
    G.json, id2idx.json, features.npy (optional)

    Arguments:
    - data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir,net_type=None):
        self.data_dir = data_dir
        self.type = net_type
        self.id2idx = None
        self.idx2id = None
        self._load_id2idx()
        self._load_G()
        self._load_features()
        #graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")
        self.load_edge_features()
        #self.feature_extract()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        if self.id2idx:
            G_data['links'] = [{'source': self.idx2id[G_data['links'][i]['source']], 'target': self.idx2id[G_data['links'][i]['target']]} for i in range(len(G_data['links']))]
        self.G = json_graph.node_link_graph(G_data)

    def _load_id2idx(self):
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        if os.path.isfile(id2idx_file):
            self.id2idx = json.load(open(id2idx_file))
            self.idx2id = {v:k for k,v in self.id2idx.items()}


    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
            self.feature_type = 'sparse'
        else:
            self.features = None


    def feature_extract(self):
        dense_feature_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.data_dir))),'profile_df.pkl')
        if os.path.isfile(dense_feature_path):
            self.feature_type = 'dense'
            with open(dense_feature_path,'rb') as file:
                profile_df = pickle.load(file)
            self.sparse_feature_name = []
            self.dense_feature_name = []
            self.dense_f_list_transforms = {}
            profile_columns = profile_df.columns
            print('loading dense features')
            for f in profile_columns:
                if f == 'username' or f == 'description':
                    continue
            ##如果特征值是一个列表，表示为密集特征，将列表中的元素合并为一个词汇表，并记录词汇表的长度
                if type(profile_df[f][0]) == list:
                    dense_f_list = profile_df[f].values.tolist()
                    vocab = []
                    for sub_list in dense_f_list:
                        for j in sub_list:
                            try:
                                vocab.append(j)
                            except:
                                print('empty feature')
                                continue
                    vocab = list(set(vocab))
                    vocab_len = len(vocab)

                    #然后，将每个列表转换为一个二进制的张量，其中每个位置表示是否存在对应的词。
                    dense_f_transform = []
                    if self.type == 's':
                        dense_f_list = dense_f_list[:len(self.G.nodes())]
                    else:
                        dense_f_list = dense_f_list[len(self.G.nodes()):]
                    for t in dense_f_list:
                        dense_f_idx = torch.zeros(1, vocab_len).long()
                        for w in t:
                            idx = vocab.index(w)
                            dense_f_idx[0, idx] = 1
                        dense_f_transform.append(dense_f_idx)
                    self.dense_f_list_transforms[f] = torch.cat(dense_f_transform, dim=0)
                    self.dense_feature_name.append({'feature_name':f})                  
                else:
                    #如果特征值不是一个列表，表示为稀疏特征，使用LabelEncoder对其进行编码，并记录特征的维度。最后，将物品特征转换为torch.Tensor类型的矩阵。
                    encoder = LabelEncoder()
                    encoder.fit(profile_df[f])
                    profile_df[f] = encoder.transform(profile_df[f])
                    feature_dim = len(encoder.classes_)
                    self.sparse_feature_name.append({'feature_name':f, 'feature_dim':feature_dim})
            self.sparse_feature_matrix = torch.from_numpy(profile_df[[f['feature_name'] for f in self.sparse_feature_name]].values)
        else:
            self.sparse_feature_name = None
            self.sparse_feature_matrix= None
            self.dense_feature_name= None
            self.dense_f_list_transforms= None

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self, sparse=False):
        return graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)
    
    def get_nodes_neighbors(self):
        return graph_utils.build_neighbors(self.G, self.id2idx)

    def get_edges(self,idx=True):
        if idx:
            return graph_utils.get_edges(self.G, self.id2idx)
        else:
            return graph_utils.get_edges(self.G)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True
    
    




def parse_args():
    parser = argparse.ArgumentParser(description="Test loading dataset")
    parser.add_argument('--source_dataset', default="/home/trunght/dataspace/graph/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="/home/trunght/dataspace/graph/douban/offline/graphsage/")
    parser.add_argument('--groundtruth', default="/home/trunght/dataspace/graph/douban/dictionaries/groundtruth")
    parser.add_argument('--output_dir', default="/home/trunght/dataspace/graph/douban/statistics/")
    return parser.parse_args()

def main(args):    
    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, "dict")
    DataPreprocess.evaluateDataset(source_dataset, target_dataset, groundtruth, args.output_dir)





if __name__ == "__main__":
    args = parse_args()
    main(args)
