from algorithms.PALE.loss import EmbeddingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

#根据给定的新旧节点ID映射关系，对data进行节点ID的重新编号，并返回编号后的data
def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long)
    row = data.edge_index[0]
    col = data.edge_index[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old

class PaleEmbedding(nn.Module):
    def __init__(self, n_nodes, embedding_dim, deg, neg_sample_size, cuda):

        """
        Parameters
        ----------
        n_nodes: int
            Number of all nodes
        embedding_dim: int
            Embedding dim of nodes
        deg: ndarray , shape = (-1,)
            Array of degrees of all nodes
        neg_sample_size : int
            Number of negative candidate to sample
        cuda: bool
            Whether to use cuda
        """

        super(PaleEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.num_nodes = n_nodes
        torch.nn.init.xavier_normal_(self.node_embedding.weight.data)
        self.fixed_data = self.node_embedding.weight.data[0]
        self.deg = deg
        self.neg_sample_size = neg_sample_size
        self.link_pred_layer = EmbeddingLossFunctions()
        self.n_nodes = n_nodes
        self.use_cuda = cuda
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss, loss0, loss1 = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        loss0 = loss0/batch_size
        loss1 = loss1/batch_size
        
        return loss, loss0, loss1

    def curvature_loss(self, walks):
        all_emb = self.node_embedding(torch.LongTensor(np.array(range(self.num_nodes))).cuda())
        # all_emb[0] = self.fixed_data
        walks_emb = all_emb[walks] # bs x wl x emb_dim
        target = walks_emb[:, 1:]
        source = walks_emb[:, :-1]
        dis = target - source
        cos_values = self.cos(dis[:, 1:], dis[:, :-1])
        loss = 1 - cos_values.mean()
        return loss


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        # node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            if self.use_cuda:
                neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            # neighbor_output = F.normalize(neighbor_output, dim=1)
            # neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output

    def get_embedding(self):
        nodes = np.arange(self.n_nodes)
        nodes = torch.LongTensor(nodes)
        if self.use_cuda:
            nodes = nodes.cuda()
        embedding = None
        BATCH_SIZE = 512
        for i in range(0, self.n_nodes, BATCH_SIZE):
            j = min(i + BATCH_SIZE, self.n_nodes)
            batch_nodes = nodes[i:j]
            if batch_nodes.shape[0] == 0: break
            batch_node_embeddings = self.forward(batch_nodes)
            if embedding is None:
                embedding = batch_node_embeddings
            else:
                embedding = torch.cat((embedding, batch_node_embeddings))

        return embedding


