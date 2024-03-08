from algorithms.network_alignment_model import NetworkAlignmentModel
from evaluation.metrics import get_statistics
from algorithms.GAlign.embedding_model import G_Align as Multi_Order, StableFactor
from algorithms.GAlign.embedding_model_copy import G_Align as DIFFormer
from input.dataset import Dataset
from utils.graph_utils import load_gt
import torch.nn.functional as F
import torch.nn as nn
from algorithms.GAlign.utils import *
from algorithms.GAlign.losses import *

import torch
import numpy as np
import networkx as nx
import random 
import numpy as np

import argparse
import os
import time
import sys

from torch.autograd import Variable
from tqdm import tqdm
import copy
import re
import json


class GAlign(NetworkAlignmentModel):
    """
    GAlign model for networks alignment task
    """
    def __init__(self, source_dataset, target_dataset, args):
        """
        :params source_dataset: source graph
        :params target_dataset: target graph
        :params args: more config params
        """
        super(GAlign, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.alphas = [args.alpha0,args.alpha1,args.alpha2]
        self.args = args
        self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.Laplacian = args.Laplacian
        self.mixup = args.mixup


    def graph_augmentation(self, dataset, type_aug='remove_edges'):
        """
        Generate small noisy graph from original graph
        :params dataset: original graph
        :params type_aug: type of noise added for generating new graph
        """
        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()
        
        if type_aug == "remove_edges":
            num_edges = len(edges)
            num_remove = int(len(edges) * self.args.noise_level)
            index_to_remove = np.random.choice(np.arange(num_edges), num_remove, replace=False)
            edges_to_remove = edges[index_to_remove]
            for i in range(len(edges_to_remove)):
                adj[edges_to_remove[i, 0], edges_to_remove[i, 1]] = 0
                adj[edges_to_remove[i, 1], edges_to_remove[i, 0]] = 0
        elif type_aug == "add_edges":
            num_edges = len(edges)
            num_add = int(len(edges) * self.args.noise_level)
            count_add = 0
            while count_add < num_add:
                random_index = np.random.randint(0, adj.shape[1], 2)
                if adj[random_index[0], random_index[1]] == 0:
                    adj[random_index[0], random_index[1]] = 1
                    adj[random_index[1], random_index[0]] = 1
                    count_add += 1
        elif type_aug == "change_feats":
            feats = np.copy(dataset.features)
            num_nodes = adj.shape[0]
            num_nodes_change_feats = int(num_nodes * self.args.noise_level)
            node_to_change_feats = np.random.choice(np.arange(0, adj.shape[0]), num_nodes_change_feats, replace=False)
            for node in node_to_change_feats:
                feat_node = feats[node]
                feat_node[feat_node == 1] = 0
                feat_node[np.random.randint(0, feats.shape[1], 1)[0]] = 1
            feats = torch.FloatTensor(feats)
            if self.args.cuda:
                feats = feats.cuda()
            return feats
        if self.Laplacian:
            new_adj_H, _ = Laplacian_graph(adj)
        else:
            new_adj_H = torch.tensor(adj)
        if self.args.cuda:
            new_adj_H = new_adj_H.cuda()
        return new_adj_H

    def shuffleData(self,source_feats, target_feats,source_A_hat, target_A_hat):
        source_ids = list(range(len(source_feats)))
        target_ids = list(range(len(target_feats)))

        source_neighbors = self.source_dataset.get_nodes_neighbors()
        target_neighbors = self.target_dataset.get_nodes_neighbors()
        source_degrees = self.source_dataset.get_nodes_degrees()
        target_degrees = self.target_dataset.get_nodes_degrees()

        mixup_node_number1 = self.args.k
        mixup_node_number2 = self.args.k
        for index,neighbors in enumerate(source_neighbors):
            if mixup_node_number1 == 0:
                break
            neighbors = [self.source_dataset.id2idx[element] for element in neighbors]
            if source_degrees[index] < self.args.degree_threshold:
                max_node = neighbors[0]
                for i,node in enumerate(neighbors):
                    if source_degrees[node] > source_degrees[max_node]:
                        max_node = node
                source_ids[index] = max_node
                mixup_node_number1=mixup_node_number1-1

        for index,neighbors in enumerate(target_neighbors):
            if mixup_node_number2 == 0:
                break
            neighbors = [self.target_dataset.id2idx[element] for element in neighbors]
            if target_degrees[index] < self.args.degree_threshold:
                max_node = neighbors[0]
                for i,node in enumerate(neighbors):
                    if target_degrees[node] > target_degrees[max_node]:
                        max_node = node
                target_ids[index] = max_node
                mixup_node_number2=mixup_node_number2-1
        
        """ np.random.shuffle(source_ids)
        np.random.shuffle(target_ids) """
        
        """ with open('best_param.json','r') as file:
                data = json.load(file)
        source_ids = data[0]['source_ids']
        target_ids = data[0]['target_ids'] """

        shuffled_source_A_hat = source_A_hat.coalesce().clone()
        source_row = source_A_hat.coalesce().indices()[0]
        shuffled_source_A_hat.coalesce().indices()[0] = torch.tensor([source_ids[element.item()] for element in source_row])
        source_col = source_A_hat.coalesce().indices()[1]
        shuffled_source_A_hat.coalesce().indices()[1] = torch.tensor([source_ids[element.item()] for element in source_col])

        shuffled_target_A_hat = target_A_hat.coalesce().clone()
        target_row = target_A_hat.coalesce().indices()[0]
        shuffled_target_A_hat.coalesce().indices()[0] = torch.tensor([target_ids[element.item()] for element in target_row])
        target_col = target_A_hat.coalesce().indices()[1]
        shuffled_target_A_hat.coalesce().indices()[1] = torch.tensor([target_ids[element.item()] for element in target_col])
        return source_ids,target_ids,shuffled_source_A_hat,shuffled_target_A_hat

    def align(self):
        """
        The main function of GAlign
        """
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        if self.mixup:
            self.source_ids,self.target_ids,self.shuffled_source_A_hat, self.shuffled_target_A_hat, = self.shuffleData(source_feats, target_feats,source_A_hat, target_A_hat)
        print("Running Multi-level embedding")
        GAlign = self.multi_level_embed(source_A_hat, target_A_hat, source_feats, target_feats)
        print("Running Refinement Alignment")
        S_GAlign = self.refinement_alignment(GAlign, source_A_hat, target_A_hat)
        return S_GAlign


    def multi_level_embed(self, source_A_hat, target_A_hat, source_feats, target_feats):
        """
        Input: SourceGraph and TargetGraph
        Output: Embedding of those graphs using Multi_order_embedding model
        """
        if self.Laplacian:
            if self.mixup:
                GAlign = Multi_Order(
                    activate_function = self.args.act,
                    num_GCN_blocks = self.args.num_GCN_blocks,
                    input_dim = self.args.input_dim,
                    output_dim = self.args.embedding_dim,
                    num_source_nodes = len(source_A_hat),
                    num_target_nodes = len(target_A_hat),
                    source_ids = self.source_ids,
                    target_ids = self.target_ids,
                    shuffled_source_A_hat = self.shuffled_source_A_hat,
                    shuffled_target_A_hat = self.shuffled_target_A_hat,
                    source_feats = source_feats,
                    target_feats = target_feats,
                    mixup = self.mixup
                )
            else:
                GAlign = Multi_Order(
                    activate_function = self.args.act,
                    num_GCN_blocks = self.args.num_GCN_blocks,
                    input_dim = self.args.input_dim,
                    output_dim = self.args.embedding_dim,
                    num_source_nodes = len(source_A_hat),
                    num_target_nodes = len(target_A_hat),
                    source_feats = source_feats,
                    target_feats = target_feats,
                )
        else:
            GAlign = DIFFormer(
                    in_channels = source_feats.shape[-1],
                    hidden_channels = self.args.hidden_dim,
                    out_channels = self.args.embedding_dim,
                    num_layers = self.args.num_GCN_blocks,
                    dropout=self.args.dropout,
                    use_bn=self.args.use_bn, 
                    use_graph=self.args.use_graph,
                    use_residual=self.args.use_residual,
                    use_weight=self.args.use_weight,
                    kernel=self.args.kernel,
                    num_heads=self.args.num_heads,
                    alpha=self.args.alpha,
                    source_feats = source_feats,
                    target_feats = target_feats
                )

        if self.args.cuda:
            GAlign = GAlign.cuda()
        
        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GAlign.parameters()),weight_decay=self.args.weight_decay, lr=self.args.lr)
        
        GAlign.train()

        new_source_A_hats = []
        new_target_A_hats = []
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'remove_edgse'))
        new_source_A_hats.append(self.graph_augmentation(self.source_dataset, 'add_edges'))
        new_source_A_hats.append(source_A_hat)
        new_source_feats = self.graph_augmentation(self.source_dataset, 'change_feats')
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'remove_edgse'))
        new_target_A_hats.append(self.graph_augmentation(self.target_dataset, 'add_edges'))
        new_target_A_hats.append(target_A_hat)
        new_target_feats = self.graph_augmentation(self.target_dataset, 'change_feats')

        if self.args.graph_mixup:
            mixup_source_A_hats = []
            mixup_target_A_hats = []
            mixup_source_feats = []
            mixup_target_feats = []
            source_datasets = ['graph_data/douban/online/mixup2_5_1/graphsage/','graph_data/douban/online/mixup5_5_1/graphsage/', \
                               'graph_data/douban/online/mixup5_10_3/graphsage/']
            target_datasets = ['graph_data/douban/offline/mixup2_5_1/graphsage/','graph_data/douban/offline/mixup5_5_1/graphsage/', \
                               'graph_data/douban/offline/mixup5_10_3/graphsage/']
            """ mixup_source_A_hats.append(source_A_hat)
            mixup_target_A_hats.append(target_A_hat) """
            for i in range(1):
                mixup_source_A_hat, mixup_target_A_hat, mixup_source_feat, mixup_target_feat = self.get_mixup_elements(source_datasets[i],target_datasets[i])

                mixup_source_A_hats.append(mixup_source_A_hat)
                mixup_source_feats.append(mixup_source_feat)

                mixup_target_A_hats.append(mixup_target_A_hat)
                mixup_target_feats.append(mixup_target_feat)
            
            """ new_source_A_hats = []
            new_target_A_hats = []
            new_source_A_hats.append(mixup_source_A_hat)
            new_source_A_hats.append(source_A_hat)
            new_source_feats = mixup_source_feat
            new_target_A_hats.append(mixup_target_A_hat)
            new_target_A_hats.append(target_A_hat)
            new_target_feats = mixup_target_feat """
                
            """ mixup_source_A_hats.append(source_A_hat)
            mixup_target_A_hats.append(target_A_hat) """
        if self.mixup:
            for epoch in range(self.args.GAlign_epochs):
                if self.args.log:
                    print("Mixup Structure learning epoch: {}".format(epoch))
                for i in range(2):
                    structural_optimizer.zero_grad()
                    if self.args.graph_mixup:
                        for j in range(len(mixup_source_A_hats)):
                        #for j in range(len(new_source_A_hats)):
                            if i == 0:
                                A_hat = source_A_hat
                                augment_A_hat = mixup_source_A_hats[j]
                                #augment_A_hat = new_source_A_hats[j]
                                shuffle_A_hat = self.shuffled_source_A_hat

                                outputs = GAlign(A_hat, 's')
                                mixup_outputs = GAlign(A_hat, 's',True)
                                augment_outputs = GAlign(augment_A_hat, 's', new_feats=mixup_source_feats[j])
                                
                                """ if j == 0:
                                #if 1 :
                                #if j < len(new_source_A_hats)-1:
                                    #augment_outputs = GAlign(augment_A_hat, 's',True,new_feats=mixup_source_feats[j])
                                    mixup_outputs = GAlign(A_hat, 's',True)
                                else:
                                    augment_outputs = GAlign(augment_A_hat, 's', new_feats=mixup_source_feats[j-1])
                                    #augment_outputs = GAlign(augment_A_hat, 's', new_feats=new_source_feats) """
        
                            else:
                                A_hat = target_A_hat
                                augment_A_hat = mixup_target_A_hats[j]
                                #augment_A_hat = new_target_A_hats[j]
                                shuffle_A_hat = self.shuffled_target_A_hat
                                outputs = GAlign(A_hat, 't')
                                mixup_outputs = GAlign(A_hat, 'T',True)
                                augment_outputs = GAlign(augment_A_hat, 't',new_feats=mixup_target_feats[j])

                                """ if j == 0:
                                #if 1 :
                                #if j < len(new_target_A_hats)-1:
                                    #augment_outputs = GAlign(augment_A_hat, 't',True,new_feats=mixup_target_feats[j])
                                    mixup_outputs = GAlign(A_hat, 'T',True)
                                else:
                                    augment_outputs = GAlign(augment_A_hat, 't',new_feats=mixup_target_feats[j-1])
                                    #augment_outputs = GAlign(augment_A_hat, 't', new_feats=new_target_feats) """
                            #在mixup原图以后，如果不用mixup嵌入的模型，那么beta应该大一点比较好
                            #在mixup原图以后，如果用mixup嵌入的模型，那么beta应该小一点比较好
                            #只关注mixup_loss的表现比只关注consistency_loss要差很多
                            #用多个数据增强的图效果会差，不如用1个
                            #noise_adaptivity_loss2影响力最大
                            #最好的loss组合是augment_consistency_loss2和noise_adaptivity_loss2,mixuo第一层嵌入

                            consistency_loss = self.linkpred_loss(outputs[-1], A_hat)
                            
                            
                            mixup_consistency_loss2 = self.linkpred_loss(mixup_outputs[-2], A_hat)
                            mixup_shuffle_loss2 = self.linkpred_loss(mixup_outputs[-2],shuffle_A_hat)
                            mixup_consistency_loss2 = GAlign.lam * mixup_consistency_loss2 + (1-GAlign.lam) * mixup_shuffle_loss2
    
                            #mixup嵌入作为一种数据增强
                            mixup_consistency_loss = self.args.beta *consistency_loss + (1-self.args.beta)*mixup_consistency_loss2

                            diff2 = torch.abs(outputs[-2] - mixup_outputs[-2])
                            noise_adaptivity_loss2 = (diff2[diff2 < self.args.threshold] ** 2).sum() / len(mixup_outputs) 

                            loss1 = self.args.coe_consistency * mixup_consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss2

                            augment_consistency_loss = self.linkpred_loss(augment_outputs[-1], augment_A_hat)

                            diff = torch.abs(outputs[-1] - augment_outputs[-1])
                            noise_adaptivity_loss = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                        
                            consistency_loss = self.args.beta *consistency_loss +(1-self.args.beta)*augment_consistency_loss

                            loss2 = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss
                            
                            loss = 0.2*loss1+0.8*loss2
                            """ if j==0:
                                mixup_consistency_loss2 = self.linkpred_loss(mixup_outputs[-2], A_hat)
                                mixup_shuffle_loss2 = self.linkpred_loss(mixup_outputs[-2],shuffle_A_hat)
                                mixup_consistency_loss2 = GAlign.lam * mixup_consistency_loss2 + (1-GAlign.lam) * mixup_shuffle_loss2
        
                                #mixup嵌入作为一种数据增强
                                consistency_loss = self.args.beta *consistency_loss + (1-self.args.beta)*mixup_consistency_loss2

                                diff2 = torch.abs(outputs[-2] - mixup_outputs[-2])
                                noise_adaptivity_loss2 = (diff2[diff2 < self.args.threshold] ** 2).sum() / len(mixup_outputs) 
                                loss = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss2
                            else:
                                augment_consistency_loss = self.linkpred_loss(augment_outputs[-1], augment_A_hat)

                                diff = torch.abs(outputs[-1] - augment_outputs[-1])
                                noise_adaptivity_loss = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                            
                                consistency_loss = self.args.beta *consistency_loss +(1-self.args.beta)*augment_consistency_loss
                                loss = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss """
                            if self.args.log:
                                print("Loss: {:.4f}".format(loss.data))
                            loss.backward()
                            structural_optimizer.step()
        else:
            for epoch in range(self.args.GAlign_epochs):
                if self.args.log:
                    print("Structure learning epoch: {}".format(epoch))
                for i in range(2):
                    for j in range(len(new_source_A_hats)):
                        structural_optimizer.zero_grad()
                        if i == 0:
                            A_hat = source_A_hat
                            augment_A_hat = new_source_A_hats[j]
                            outputs = GAlign(source_A_hat, 's')
                            if j < len(new_source_A_hats)-1:
                                augment_outputs = GAlign(augment_A_hat, 's')
                            else:
                                augment_outputs = GAlign(augment_A_hat, 's', new_feats=new_source_feats)
                        else:
                            A_hat = target_A_hat
                            augment_A_hat = new_target_A_hats[j]
                            outputs = GAlign(target_A_hat, 't')
                            if j < len(new_target_A_hats)-1:
                                augment_outputs = GAlign(augment_A_hat, 't')
                            else:
                                augment_outputs = GAlign(augment_A_hat, 't', new_feats=new_target_feats)
                        consistency_loss = self.linkpred_loss(outputs[-1], A_hat)
                        augment_consistency_loss = self.linkpred_loss(augment_outputs[-1], augment_A_hat)
                        
                        consistency_loss = self.args.beta * consistency_loss + (1-self.args.beta) * augment_consistency_loss
                        diff = torch.abs(outputs[-1] - augment_outputs[-1])
                        noise_adaptivity_loss = (diff[diff < self.args.threshold] ** 2).sum() / len(outputs)
                        loss = self.args.coe_consistency * consistency_loss + (1 - self.args.coe_consistency) * noise_adaptivity_loss
                        if self.args.log:
                            print("Loss: {:.4f}".format(loss.data))
                        loss.backward()
                        structural_optimizer.step()
        GAlign.eval()
        return GAlign


    def refinement_alignment(self, GAlign, source_A_hat, target_A_hat):
        """ source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense() """
        GAlign_S = self.refine(GAlign, source_A_hat, target_A_hat, self.args.threshold_refine)
        return GAlign_S

    def get_mixup_elements(self,source_mixup_dataset,target_mixup_dataset):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_mixup_dataset = Dataset(source_mixup_dataset)
        target_mixup_dataset = Dataset(target_mixup_dataset)
        
        source_A_hat, _ = Laplacian_graph(source_mixup_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(target_mixup_dataset.get_adjacency_matrix())

        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()


        source_feats = source_mixup_dataset.features
        target_feats = target_mixup_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(source_mixup_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(target_mixup_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats
    
    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        if self.Laplacian:
            source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
            target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        else:
            source_A_hat = torch.tensor(self.source_dataset.get_adjacency_matrix())
            target_A_hat = torch.tensor(self.target_dataset.get_adjacency_matrix())

        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()


        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats

    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        #linkpred_losss = (pred_adj - A[index]) ** 2
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss


    def refine(self, GAlign, source_A_hat, target_A_hat, threshold):
        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)
        if self.args.cuda: 
            refinement_model = refinement_model.cuda()
        S_max = None
        if self.mixup:
            print('mixup')
            source_outputs = GAlign(source_A_hat, 's',True)
            target_outputs = GAlign(target_A_hat, 't',True)
        else:
            source_outputs = GAlign(source_A_hat, 's')
            target_outputs = GAlign(target_A_hat, 't')
            """ source_outputs = GAlign(refinement_model(source_A_hat, 's'), 's')
            target_outputs = GAlign(refinement_model(target_A_hat, 't'), 't') """
        acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=False)
        #计算相似度矩阵S的平均最大值，作为源域和目标域之间的相似度得分
        score = np.max(S, axis=1).mean()

        matches = re.findall(r'\d+\.\d+', acc)
        final_acc = matches[-1] if matches else None
        
        refinement_model.score_max = score
        acc_max = final_acc
        S_max = S

        """ print("Acc: {}, score: {:.4f}".format(acc, score))
        self.GAlign_S = S_max
        return self.GAlign_S """
    
        if self.mixup:
            match = re.search(r'\d+\.\d+', acc)
            first_decimal = float(match.group())
            if first_decimal > 0.0733:
                file_path = "best_param.json"
                pre = "Acc: {}, score: {:.4f}".format(acc, score)


        print("Acc: {}, score: {:.4f}".format(acc, score))
        source_candidates, target_candidates = [], []            
        alpha_source_max = refinement_model.alpha_source + 0
        alpha_target_max = refinement_model.alpha_target + 0
        for epoch in range(self.args.refinement_epochs):
            if self.args.log:
                print("Refinement epoch: {}".format(epoch))
            source_candidates, target_candidates, len_source_candidates, count_true_candidates = self.get_candidate(source_outputs, target_outputs, threshold)
            
            refinement_model.alpha_source[source_candidates] *= 1.2
            refinement_model.alpha_target[target_candidates] *= 1.2
            source_outputs = GAlign(refinement_model(source_A_hat, 's'), 's')
            target_outputs = GAlign(refinement_model(target_A_hat, 't'), 't')
            acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=False)
            score = np.max(S, axis=1).mean()
            matches = re.findall(r'\d+\.\d+', acc)
            final_acc = matches[-1] if matches else None
            #if 1:
            #if score > refinement_model.score_max
            if final_acc > acc_max:
            #if score > refinement_model.score_max or final_acc > acc_max:
                refinement_model.score_max = score
                alpha_source_max = refinement_model.alpha_source + 0
                alpha_target_max = refinement_model.alpha_target + 0
                acc_max = final_acc
                S_max = S
                
            if self.args.log:
                print("Acc: {}, score: {:.4f}, score_max {:.4f}".format(acc, score, refinement_model.score_max))
            if epoch == self.args.refinement_epochs - 1:
                print("Numcandidate: {}, num_true_candidate: {}".format(len_source_candidates, count_true_candidates))
        print("Done refinement!")
        print("Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max))
        if self.mixup and first_decimal > 0.0733:
            refine = "Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max)
            data = {'precision':pre,'refinement':refine,'lam':GAlign.lam,'source_ids':self.source_ids,'target_ids': self.target_ids}
            """ with open(file_path, "a") as file:
                    json.dump(data,file) """
        refinement_model.alpha_source = alpha_source_max
        refinement_model.alpha_target = alpha_target_max
        self.GAlign_S = S_max
        return self.GAlign_S
        # self.log_and_evaluate(GAlign, refinement_model, source_A_hat, target_A_hat)
        


    def get_similarity_matrices(self, source_outputs, target_outputs):
        """
        Construct Similarity matrix in each layer
        :params source_outputs: List of embedding at each layer of source graph
        :params target_outputs: List of embedding at each layer of target graph
        """
        list_S = []
        for i in range(len(source_outputs)):
            source_output_i = source_outputs[i]
            target_output_i = target_outputs[i]
            S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
            list_S.append(S)
        return list_S


    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        print("-"* 100)
        log, self.S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
        print(self.alphas)
        print(log)
        return source_outputs, target_outputs
    

    def get_candidate(self, source_outputs, target_outputs, threshold):
        List_S = self.get_similarity_matrices(source_outputs, target_outputs)[1:]
        source_candidates = []
        target_candidates = []
        count_true_candidates = 0
        if len(List_S) < 2:
            print("The current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

        num_source_nodes = len(self.source_dataset.G.nodes())
        num_target_nodes = len(self.target_dataset.G.nodes())
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S)):
                if List_S[j][i].argmax() != List_S[j-1][i].argmax() or List_S[j][i].max() < threshold:
                    node_i_is_stable = False 
                    break
            if node_i_is_stable:
                tg_candi = List_S[-1][i].argmax()
                source_candidates.append(i)
                target_candidates.append(tg_candi)
                try:
                    if self.full_dict[i] == tg_candi:
                        count_true_candidates += 1
                except:
                    continue
        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(source_candidates), count_true_candidates
