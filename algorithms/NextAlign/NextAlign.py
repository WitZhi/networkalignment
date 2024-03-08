import numpy as np
import dgl
from copy import deepcopy

from algorithms.network_alignment_model import NetworkAlignmentModel

from utils.graph_utils import load_gt
import torch
from algorithms.NextAlign.model import Model
from algorithms.NextAlign.negative_sampling import negative_sampling_exact
from algorithms.NextAlign.node2vec import load_walks
from algorithms.NextAlign.utils import *
from algorithms.NextAlign.rwr_scoring import rwr_scores
from algorithms.NextAlign.data import Train_Data
from algorithms.NextAlign.test import test

from torch.utils.data import DataLoader

import time
import os
import pdb

class NextAlign(NetworkAlignmentModel):


    def __init__(self, source_dataset, target_dataset,args):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.args = args

    def align(self):
        args = self.args
        G1,G2 = preprocessing(self.source_dataset.G,self.target_dataset.G)
        #G1 = self.source_dataset.G
        #G2 = self.target_dataset.G
        anchor_links = load_gt(args.train_dict,self.source_dataset.id2idx,self.target_dataset.id2idx,format='dict')
        anchor_links = np.array([[key,value] for key,value in anchor_links.items()])
        anchor_nodes1, anchor_nodes2 = anchor_links[:, 0], anchor_links[:, 1]
        anchor_links2 = anchor_nodes2
        test_pairs = load_gt(args.test_dict,format='dict')
        test_pairs = np.array([[key,value] for key,value in test_pairs.items()])
        datasets = args.source_dataset.split('/')[1]
        x1 = self.source_dataset.features
        x2 = self.target_dataset.features
        edge_index1 = self.source_dataset.get_edges()
        edge_index2 = self.target_dataset.get_edges()
        

        # run node2vec or load from existing file for positive context pairs
        t0 = time.time()
        if not os.path.isfile('graph_data/%s/node2vec_context_pairs_%s_%.1f.npz' % (datasets,datasets, args.ratio)):
            # run node2vec from scratch
            walks1 = load_walks(G1, args.p, args.q, args.num_walks, args.walk_length)
            walks2 = load_walks(G2, args.p, args.q, args.num_walks, args.walk_length)
            context_pairs1 = extract_pairs(walks1, anchor_nodes1)
            context_pairs2 = extract_pairs(walks2, anchor_nodes2)
            context_pairs1, context_pairs2 = balance_inputs(context_pairs1, context_pairs2)
            np.savez('graph_data/%s/node2vec_context_pairs_%s_%.1f.npz' % (datasets,datasets, args.ratio), context_pairs1=context_pairs1, context_pairs2=context_pairs2)
        else:
            contexts = np.load('graph_data/%s/node2vec_context_pairs_%s_%.1f.npz' % (datasets,datasets, args.ratio))
            context_pairs1 = contexts['context_pairs1']
            context_pairs2 = contexts['context_pairs2']
        print('Finished positive context pair sampling in %.2f seconds' % (time.time() - t0))

        # run random walk with restart or load from existing file for pre-positioning
        t0 = time.time()
        if not os.path.isfile('graph_data/%s/rwr_emb_%s_%.1f.npz' % (datasets,datasets, args.ratio)):
            #anchor_links原本是一个(n,2)的多维数组，现在变成了(n)维的字典
            rwr_score1, rwr_score2 = rwr_scores(G1, G2, anchor_links)
            np.savez('graph_data/%s/rwr_emb_%s_%.1f.npz' % (datasets,datasets, args.ratio), rwr_score1=rwr_score1, rwr_score2=rwr_score2)
        else:
            scores = np.load('graph_data/%s/rwr_emb_%s_%.1f.npz' % (datasets,datasets, args.ratio))
            rwr_score1, rwr_score2 = scores['rwr_score1'], scores['rwr_score2']

        # Set initial relative positions
        position_score1, position_score2 = anchor_emb(G1, G2, anchor_links)
        print(G1.nodes())
        for node in G1.nodes():
            if node not in anchor_nodes1:
                position_score1[node] += rwr_score1[node]
        for node in G2.nodes():
            if node not in anchor_nodes2:
                position_score2[node] += rwr_score2[node]
        x1 = (position_score1, x1) if args.use_attr else position_score1
        x2 = (position_score2, x2) if args.use_attr else position_score2
        print('Finished initial relative positioning in %.2f seconds' % (time.time() - t0))

        # merge input networks into a world-view network
        t0 = time.time()
        node_mapping1 = np.arange(G1.number_of_nodes()).astype(np.int64)
        edge_index, edge_types, x, node_mapping2 = merge_graphs(edge_index1, edge_index2, x1, x2, anchor_links)
        print('Finished merging networks in %.2f seconds' % (time.time() - t0))

        # input node features: (one-hot encoding, position, optional - node attributes)
        x1 = np.arange(len(x[0]), dtype=np.int64) if args.use_attr else np.arange(len(x), dtype=np.int64)
        x2 = x[0].astype(np.float32) if args.use_attr else x.astype(np.float32)
        x = (x1, x2, x[1]) if args.use_attr else (x1, x2)

        landmark = torch.from_numpy(anchor_nodes1)
        num_nodes = x[0].shape[0]
        num_attrs = x[2].shape[1] if args.use_attr else 0
        num_anchors = x[1].shape[1]
        
        model = Model(num_nodes, args.dim, landmark, args.dist, num_anchors=num_anchors, num_attrs=num_attrs)

        if args.cuda:
            # to device
            landmark = landmark.cuda()
            model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            g = dgl.graph((edge_index.T[0], edge_index.T[1]),device='cuda:%d' % args.device)
            x1 = torch.from_numpy(x[0]).cuda()
            x2 = torch.from_numpy(x[1]).cuda()
            x = (x1, x2, torch.from_numpy(x[2]).cuda()) if args.use_attr else (x1, x2)
            edge_types = torch.from_numpy(edge_types).cuda()
            node_mapping1 = torch.from_numpy(node_mapping1).cuda()
            node_mapping2 = torch.from_numpy(node_mapping2).cuda()

        ################################################################################################
        # prepare training data
        dataset = Train_Data(context_pairs1, context_pairs2)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_size = len(data_loader)
        print(data_loader_size)

        ################################################################################################
        # start training
        pn_examples1, pn_examples2, pnc_examples1, pnc_examples2 = [], [], [], []
        t_neg_sampling, t_get_emb, t_loss, t_model = 0, 0, 0, 0
        total_loss = 0

        topk = [1, 10, 30, 50, 100]
        max_hits = np.zeros(len(topk), dtype=np.float32)
        max_hit_10, max_hit_30, max_epoch = 0, 0, 0


        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(data_loader):
                nodes1, nodes2 = data
                nodes1 = nodes1.cuda()
                nodes2 = nodes2.cuda()
                anchor_nodes1 = nodes1[:, 0].reshape((-1,))
                pos_context_nodes1 = nodes1[:, 1].reshape((-1,))
                anchor_nodes2 = nodes2[:, 0].reshape((-1,))
                pos_context_nodes2 = nodes2[:, 1].reshape((-1,))
                # forward pass
                t0 = time.time()
                out_x = model(g, x, edge_types)
                t_model += (time.time() - t0)

                t0 = time.time()
                context_pos1_emb = out_x[node_mapping1[pos_context_nodes1]]
                context_pos2_emb = out_x[node_mapping2[pos_context_nodes2]]

                pn_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                                'p_n', 'g1')
                pn_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                                'p_n', 'g2')
                pnc_examples1, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes1, node_mapping1,
                                                                    'p_nc', 'g1', node_mapping2=node_mapping2)
                pnc_examples2, _ = negative_sampling_exact(out_x, args.N_negs, anchor_nodes2, node_mapping2,
                                                                    'p_nc', 'g2', node_mapping2=node_mapping1)

                t_neg_sampling += (time.time() - t0)

                # get node embeddings
                t0 = time.time()

                pn_examples1 = torch.from_numpy(pn_examples1).reshape((-1,)).cuda()
                pn_examples2 = torch.from_numpy(pn_examples2).reshape((-1,)).cuda()
                pnc_examples1 = torch.from_numpy(pnc_examples1).reshape((-1,)).cuda()
                pnc_examples2 = torch.from_numpy(pnc_examples2).reshape((-1,)).cuda()

                anchor1_emb = out_x[node_mapping1[anchor_nodes1]]
                anchor2_emb = out_x[node_mapping2[anchor_nodes2]]
                context_neg1_emb = out_x[node_mapping1[pn_examples1]]
                context_neg2_emb = out_x[node_mapping2[pn_examples2]]
                anchor_neg1_emb = out_x[node_mapping2[pnc_examples1]]
                anchor_neg2_emb = out_x[node_mapping1[pnc_examples2]]

                input_embs = (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb,
                            context_neg2_emb, anchor_neg1_emb, anchor_neg2_emb)

                t_get_emb += (time.time() - t0)

                # compute loss
                t0 = time.time()
                loss1, loss2 = model.loss(input_embs)
                total_loss = args.coeff1 * loss1 + args.coeff2 * loss2
                t_loss += (time.time() - t0)

                print("Epoch:{}, Iteration:{}, Training loss:{}, Loss1:{},"
                    " Loss2:{}".format(epoch + 1, i + 1, round(total_loss.item(), 4), round(loss1.item(), 4),
                                        round(loss2.item(), 4)))

                # backward pass
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_t_model = round(t_model / ((epoch+1) * data_loader_size), 2)
            avg_t_neg_sampling = round(t_neg_sampling / ((epoch+1) * data_loader_size), 2)
            avg_t_get_emb = round(t_get_emb / ((epoch+1) * data_loader_size), 2)
            avg_t_loss = round(t_loss / ((epoch+1) * data_loader_size), 2)
            time_cost = [avg_t_model, avg_t_neg_sampling, avg_t_get_emb, avg_t_loss]

            train_hits,S_train = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, anchor_links, anchor_links2, args.dist)
            hits,S_test = test(model, topk, g, x, edge_types, node_mapping1, node_mapping2, test_pairs, anchor_links2, args.dist, 'testing')
            print("Epoch:{}, Training loss:{}, Train_Hits:{},  Test_Hits:{}, Time:{}".format(
                epoch+1, round(total_loss.item(), 4), train_hits, hits, time_cost))

            if hits[2] > max_hit_30 or (hits[2] == max_hit_30 and hits[1] > max_hits[1]):
                max_hit_30 = hits[2]
                max_hits = hits
                max_epoch = epoch + 1

            print("Max test hits:{} at epoch: {}".format(max_hits, max_epoch))

        if args.use_attr:
            with open('results/results_%s_attr_%.1f.txt' % (datasets, args.ratio), 'a+') as f:
                f.write(', '.join([str(x) for x in max_hits]) + '\n')
        else:
            with open('results/results_%s_%.1f.txt' % (datasets, args.ratio), 'a+') as f:
                f.write(', '.join([str(x) for x in max_hits]) + '\n')
        return S_test