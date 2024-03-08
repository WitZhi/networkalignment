from input.dataset import Dataset
from time import time
from algorithms import *
from evaluation.metrics import get_statistics

import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="graph_data/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="graph_data/douban/offline/graphsage/")
    parser.add_argument('--groundtruth',    default="graph_data/douban/dictionaries/groundtruth")
    """ parser.add_argument('--source_dataset', default="graph_data/allmv_tmdb/allmv/graphsage/")
    parser.add_argument('--target_dataset', default="graph_data/allmv_tmdb/tmdb/graphsage/") 
    parser.add_argument('--groundtruth',    default="graph_data/allmv_tmdb/dictionaries/groundtruth") """
    parser.add_argument('--seed',           default=145,    type=int)
    parser.add_argument('--device',           default=2,    type=int)
    subparsers = parser.add_subparsers(dest="algorithm", help='Choose 1 of the algorithm from: IsoRank, FINAL, REGAL, IONE, PALE,CENALP,GAlign,MGLAlign,GradAlignPlus,NAME')
    

    # IsoRank
    parser_isorank = subparsers.add_parser('IsoRank', help='IsoRank algorithm')
    parser_isorank.add_argument('--H',                   default=None, help="Priority matrix")
    parser_isorank.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_isorank.add_argument('--alpha',               default=0.82, type=float)
    parser_isorank.add_argument('--tol',                 default=1e-8, type=float)
    parser_isorank.add_argument('--train_dict', default="graph_data/douban/dictionaries/node,split=0.8.train.dict", type=str)


    # FINAL
    parser_final = subparsers.add_parser('FINAL', help='FINAL algorithm')
    parser_final.add_argument('--H',                   default=None, help="Priority matrix")
    parser_final.add_argument('--max_iter',            default=30, type=int, help="Max iteration")
    parser_final.add_argument('--alpha',               default=0.6, type=float)
    parser_final.add_argument('--tol',                 default=1e-2, type=float)
    parser_final.add_argument('--train_dict', default='', type=str)


    # IONE
    parser_ione = subparsers.add_parser('IONE', help='IONE algorithm')
    parser_ione.add_argument('--train_dict', default="groundtruth.train", help="Groundtruth use to train.")
    parser_ione.add_argument('--epochs', default=60, help="Total iterations.", type=int)
    parser_ione.add_argument('--dim', default=100, help="Embedding dimension.")
    parser_ione.add_argument('--cuda', action='store_true')
    parser_ione.add_argument('--lr', type=float, default=0.08)


    # REGAL
    parser_regal = subparsers.add_parser('REGAL', help='REGAL algorithm')
    parser_regal.add_argument('--attrvals', type=int, default=2,
                        help='Number of attribute values. Only used if synthetic attributes are generated')
    parser_regal.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser_regal.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')
    parser_regal.add_argument('--max_layer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser_regal.add_argument('--alpha', type=float, default=0.01, help="Discount factor for further layers")
    parser_regal.add_argument('--gammastruc', type=float, default=1, help="Weight on structural similarity")
    parser_regal.add_argument('--gammaattr', type=float, default=1, help="Weight on attribute similarity")
    parser_regal.add_argument('--num_top', type=int, default=10,
                        help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser_regal.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")

    # PALE
    parser_PALE = subparsers.add_parser('PALE', help="PALE algorithm")
    parser_PALE.add_argument('--cuda',                action='store_true')

    parser_PALE.add_argument('--learning_rate1',      default=0.001,        type=float)
    parser_PALE.add_argument('--embedding_dim',       default=300,         type=int)
    parser_PALE.add_argument('--batch_size_embedding',default=512,         type=int)
    parser_PALE.add_argument('--embedding_epochs',    default=500,        type=int)
    parser_PALE.add_argument('--neg_sample_size',     default=10,          type=int)
    parser_PALE.add_argument('--num_walks',     default=10,          type=int)
    parser_PALE.add_argument('--walk_len',     default=10,          type=int)
    parser_PALE.add_argument('--cur_weight',     default=1,          type=float)
    

    parser_PALE.add_argument('--learning_rate2',      default=0.001,       type=float)
    parser_PALE.add_argument('--batch_size_mapping',  default=32,         type=int)
    parser_PALE.add_argument('--mapping_epochs',      default=100,         type=int)
    parser_PALE.add_argument('--mapping_model',       default='linear')
    parser_PALE.add_argument('--activate_function',   default='sigmoid')
    parser_PALE.add_argument('--toy',   action="store_true")
    parser_PALE.add_argument('--train_dict',          default='graph_data/douban/dictionaries/node,split=0.2.train.dict')
    parser_PALE.add_argument('--embedding_name',          default='')
    


    # CENALP
    parser_CENALP = subparsers.add_parser('CENALP', help="CENALP algorithm")
    parser_CENALP.add_argument('--cuda',                action='store_true')

    parser_CENALP.add_argument('--embedding_dim',       default=64,         type=int)
    parser_CENALP.add_argument('--num_walks',       default=20,         type=int)
    parser_CENALP.add_argument('--neg_sample_size',       default=10,         type=int)
    parser_CENALP.add_argument('--walk_len',       default=5,         type=int)
    parser_CENALP.add_argument('--alpha', default=5, type=float)
    parser_CENALP.add_argument('--switch_prob', default=0.3, type=float)
    parser_CENALP.add_argument('--batch_size', default=512, type=int)
    parser_CENALP.add_argument('--walk_every', default=8, type=int)
    parser_CENALP.add_argument('--learning_rate', default=0.01, type=float)
    parser_CENALP.add_argument('--threshold', default=0.5, type=float)
    parser_CENALP.add_argument('--train_dict', default="graph_data/douban/dictionaries/node,split=0.2.train.dict", type=str)
    parser_CENALP.add_argument('--num_sample',     default=300,          type=int, help="Number of samples for linkprediction")
    parser_CENALP.add_argument('--linkpred_epochs',     default=10,          type=int, help="Number of linkprediction epochs")
    parser_CENALP.add_argument('--num_iteration_epochs',     default=15,          type=int, help="Number of pair to add each epoch")



    # GAlign
    parser_GAlign = subparsers.add_parser("GAlign", help="GAlign algorithm")
    parser_GAlign.add_argument('--cuda',default=True,action="store_true")
    parser_GAlign.add_argument('--act', type=str, default='tanh')
    parser_GAlign.add_argument('--log',default=False, action="store_true", help="Just to print loss")
    parser_GAlign.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_GAlign.add_argument('--input_dim', default=100, help="Just ignore it")

    parser_GAlign.add_argument('--alpha0', type=float, default=0)
    parser_GAlign.add_argument('--alpha1', type=float, default=0)
    parser_GAlign.add_argument('--alpha2', type=float, default=1)
    parser_GAlign.add_argument('--Laplacian',default=1,type=int,help='是否使用图的拉普拉斯矩阵,1是原本的')
    

    parser_GAlign.add_argument('--embedding_dim',       default=200,         type=int)
    parser_GAlign.add_argument('--GAlign_epochs',    default=20,        type=int)
    parser_GAlign.add_argument('--lr', default=0.01, type=float)
    parser_GAlign.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_GAlign.add_argument('--hidden_dim',       default=200,         type=int)
    parser_GAlign.add_argument('--num_heads',         default=1, type=int)
    parser_GAlign.add_argument('--kernel', default='sigmoid', type=str,help='sigmoid and simple')
    parser_GAlign.add_argument('--dropout',         default=0, type=int)
    parser_GAlign.add_argument('--use_bn',         default=False, type=bool)
    parser_GAlign.add_argument('--use_graph',         default=True, type=bool)
    parser_GAlign.add_argument('--use_residual',         default=False, type=bool)
    parser_GAlign.add_argument('--use_weight',         default=False, type=bool)
    parser_GAlign.add_argument('--alpha',               default=0.5, type=float)

    # refinement
    parser_GAlign.add_argument('--refinement_epochs', default=10, type=int)
    parser_GAlign.add_argument('--threshold_refine', type=float, default=0.94, help="The threshold value to get stable candidates")

    parser_GAlign.add_argument('--weight_decay',         default=1e-4, type=int)
    parser_GAlign.add_argument('--noise_level', type=float, default=0.1)
    # loss
    parser_GAlign.add_argument('--beta', type=float, default=0.8, help='balancing source-target and source-augment')
    parser_GAlign.add_argument('--threshold', type=float, default=0.01, help='confidence threshold for adaptivity loss')
    parser_GAlign.add_argument('--coe_consistency', type=float, default=0.8, help='balancing consistency and adaptivity loss')

    # NAME
    parser_NAME = subparsers.add_parser("NAME", help="NAME algorithm")
    parser_NAME.add_argument('--cuda',                action="store_true")
    parser_NAME.add_argument('--embedding_dim',       default=200,         type=int)
    parser_NAME.add_argument('--NAME_epochs',    default=20,        type=int)
    parser_NAME.add_argument('--lr', default=0.01, type=float)
    parser_NAME.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_NAME.add_argument('--act', type=str, default='tanh')
    parser_NAME.add_argument('--log', action="store_true", help="Just to print loss")
    parser_NAME.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_NAME.add_argument('--input_dim', default=100, help="Just ignore it")
    parser_NAME.add_argument('--train_dict', type=str)
    parser_NAME.add_argument('--alpha0', type=float, default=1)
    parser_NAME.add_argument('--alpha1', type=float, default=1)
    parser_NAME.add_argument('--alpha2', type=float, default=1)
    parser_NAME.add_argument('--source_embedding')
    parser_NAME.add_argument('--target_embedding')

    # refinement
    parser_NAME.add_argument('--refinement_epochs', default=10, type=int)
    parser_NAME.add_argument('--refine', action="store_true", help="wheather to use refinement step")
    parser_NAME.add_argument('--threshold_refine', type=float, default=0.94, help="The threshold value to get stable candidates")
    # augmentation, let noise_level = 0 if dont want to use it
    parser_NAME.add_argument('--noise_level', default=0.001, type=float, help="noise to add to augment graph")
    parser_NAME.add_argument('--coe_consistency', default=0.2, type=float, help="consistency weight")
    parser_NAME.add_argument('--threshold', default=0.01, type=float, 
                    help="Threshold of for sharpenning")
    parser_NAME.add_argument('--embedding_name',          default='')
    parser_NAME.add_argument('--pale_emb_lr',    type=float,      default=0.01)
    parser_NAME.add_argument('--pale_map_lr',    type=float,      default=0.01)
    parser_NAME.add_argument('--pale_emb_epochs',    type=int,      default=500)
    parser_NAME.add_argument('--pale_map_epochs',    type=int,      default=500)
    parser_NAME.add_argument('--pale_emb_batchsize',    type=int,      default=512)
    parser_NAME.add_argument('--num_parts',    type=int,      default=8)
    parser_NAME.add_argument('--mincut_lr',    type=float,      default=0.001)
    parser_NAME.add_argument('--temp',    type=float,      default=1)
    parser_NAME.add_argument('--mincut_epochs',    type=int,      default=2000)
    parser_NAME.add_argument('--hard',    action='store_true')
    parser_NAME.add_argument('--beta',    type=float,      default=1)
    parser_NAME.add_argument('--balance_node',    action='store_true')
    parser_NAME.add_argument('--lam',    type=float,      default=0.99999)
    parser_NAME.add_argument('--w2v_lam',    type=float,      default=0)
    parser_NAME.add_argument('--new',    action='store_true')
    parser_NAME.add_argument('--clip',    type=float,      default=2.0)
    parser_NAME.add_argument('--anneal',    action='store_true')
    parser_NAME.add_argument('--min_temp',    type=float,      default=0.1)
    parser_NAME.add_argument('--debug',    action='store_true')
    parser_NAME.add_argument('--file',    type=str, default = None)

    #Grad-Align-Plus
    parser_GradAlignPlus = subparsers.add_parser('GradAlignPlus', help='GradAlignFull algorithm')
    parser_GradAlignPlus.add_argument('--data_folder', nargs='?', default='dataset/graph/')
    parser_GradAlignPlus.add_argument('--alignment_folder', nargs='?', default='dataset/alignment/',
                         help="Make sure the alignment numbering start from 0")
    parser_GradAlignPlus.add_argument('--layers', help='神经网络的层数',type=int, default=3)  
    parser_GradAlignPlus.add_argument('--hid_dim', type=int,nargs='?', default=100) 
    #parser_GradAlignPlus.add_argument('--train_ratio', nargs='?', default= 0.1) 
    parser_GradAlignPlus.add_argument('--train_type',default='super') 
    parser_GradAlignPlus.add_argument("--epochs", type=int, default=20)
    #有监督版本
    parser_GradAlignPlus.add_argument('--train_dict', default='graph_data/douban/dictionaries/node,split=0.2.train.dict',type=str)
    parser_GradAlignPlus.add_argument('--test_dict',default='graph_data/douban/dictionaries/node,split=0.2.test.dict', type=str)
    parser_GradAlignPlus.add_argument('--graphname', nargs='?', default='fb-tt') 
    parser_GradAlignPlus.add_argument('--centrality', nargs='?', default='eigenvector') 
    parser_GradAlignPlus.add_argument('--mode', nargs='?', default='not_perturbed', help="not_perturbed or perturbed") 
    parser_GradAlignPlus.add_argument('--edge_portion', nargs='?', default=0.05,  help="a param for the perturbation case")
    parser_GradAlignPlus.add_argument('--att_portion', nargs='?', default=0,  help="a param for the perturbation case")  

    parser_GradAlignPlus.add_argument('--cuda',action='store_true')
    parser_GradAlignPlus.add_argument('--aug',type=bool,default=False)

    #学习率和权重衰减
    parser_GradAlignPlus.add_argument("--local_lr", type=float, default=0.01,help="local_lr就是alpha,用来快速进行梯度下降的，对应公式seta'=seta-alpha*seta")
    parser_GradAlignPlus.add_argument("--lr", type=float, default=0.001)
    parser_GradAlignPlus.add_argument("--weight_decay", type=float, default=0.01)
    parser_GradAlignPlus.add_argument('--alpha0', type=float, default=1)
    parser_GradAlignPlus.add_argument('--alpha1', type=float, default=0.8)
    parser_GradAlignPlus.add_argument('--alpha2', type=float, default=0.5)
    parser_GradAlignPlus.add_argument('--alpha3', type=float, default=0.5)
    parser_GradAlignPlus.add_argument('--alpha4', type=float, default=0.5)
    parser_GradAlignPlus.add_argument('--alpha5', type=float, default=0.5)
    parser_GradAlignPlus.add_argument('--alpha6', type=float, default=0.5)

    #NextAlign
    parser_NextAlign = subparsers.add_parser('NextAlign', help='NextAlign algorithm')
    parser_NextAlign.add_argument('--train_type',default='super',type=str,help='unsuper and super and self-super')
    parser_NextAlign.add_argument('--cuda',action='store_true')
    parser_NextAlign.add_argument('--train_dict', default='graph_data/douban/dictionaries/node,split=0.2.train.dict',type=str)
    parser_NextAlign.add_argument('--test_dict',default='graph_data/douban/dictionaries/node,split=0.2.test.dict', type=str)
    parser_NextAlign.add_argument('--dim', type=int, default=128, help='dimension of output embeddings.')
    parser_NextAlign.add_argument('--num_layer', type=int, default=1, help='number of layers.')
    parser_NextAlign.add_argument('--ratio', type=float, default=0.2, help='training ratio.')
    parser_NextAlign.add_argument('--coeff1', type=float, default=1.0, help='coefficient for within-network link prediction loss.')
    parser_NextAlign.add_argument('--coeff2', type=float, default=1.0, help='coefficient for anchor link prediction loss.')
    parser_NextAlign.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser_NextAlign.add_argument('--epochs', type=int, default=1, help='maximum number of epochs.')
    parser_NextAlign.add_argument('--batch_size', type=int, default=2000, help='batch_size.')
    parser_NextAlign.add_argument('--walks_num', type=int, default=100,
                            help='length of walk per user node.')
    parser_NextAlign.add_argument('--N_steps', type=int, default=10,
                            help='burn-in iteration.')
    parser_NextAlign.add_argument('--N_negs', type=int, default=20,
                            help='number of negative samples per anchor node.')
    parser_NextAlign.add_argument('--p', type=int, default=1,
                            help='return hyperparameter. Default is 1.')
    parser_NextAlign.add_argument('--q', type=int, default=1,
                            help='inout hyperparameter. Default is 1.')
    parser_NextAlign.add_argument('--walk_length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser_NextAlign.add_argument('--num_walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser_NextAlign.add_argument('--use_attr', action='store_true')
    parser_NextAlign.add_argument('--dist', type=str, default='L1', help='distance for scoring.')

    #MGLAlign
    parser_MGLAlign = subparsers.add_parser('MGLAlign', help='MGLAlign algorithm')

    parser_MGLAlign.add_argument('--train_type',default='unsuper',type=str,help='unsuper and super and self-super')
    parser_MGLAlign.add_argument('--cuda',action='store_true')

    #有监督版本
    parser_MGLAlign.add_argument('--train_dict', default='graph_data/douban/dictionaries/node,split=0.2.train.dict',type=str)
    parser_MGLAlign.add_argument('--test_dict',default='graph_data/douban/dictionaries/node,split=0.2.test.dict', type=str)
    #训练参数
    parser_MGLAlign.add_argument("--batch_size", type=int, default=128)
    parser_MGLAlign.add_argument("--epochs", type=int, default=30)

    #神经网络的层数和维度
    parser_MGLAlign.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_MGLAlign.add_argument("--GNN_hidden_dim", type=int, default=256,help='GNN的隐藏层维度')
    #以下两个需要相同，不然就要加线性层
    parser_MGLAlign.add_argument("--GCN_output_dim", type=int, default=64,help='节点经过GCN嵌入结构后的维度')
    parser_MGLAlign.add_argument("--embedding_decode_dim", type=int, default=64,help='辅助嵌入经过解码器还原后的维度')

    parser_MGLAlign.add_argument("--feature_embedding_dim", type=int, default=8,help='稀疏节点属性进行嵌入后的维度')
    parser_MGLAlign.add_argument("--dense_embedding_dim", type=int, default=16,help='密集节点属性进行嵌入后的维度')
    parser_MGLAlign.add_argument("--MLP_hidden_dim", type=int, default=256,help='MLP的隐藏层维度')
    parser_MGLAlign.add_argument("--MLP_output_dim", type=int, default=256,help='MLP的输出层维度')
    parser_MGLAlign.add_argument('--act', type=str, default='tanh',help='神经网络层之间的非线性激活函数,other:sigmoid,relu')

    #跨图随机游走的嵌入
    #parser_MGLAlign.add_argument('--embedding_dim',       default=64,         type=int)
    parser_MGLAlign.add_argument('--num_walks',       default=5,         type=int)
    parser_MGLAlign.add_argument('--neg_sample_size',       default=10,         type=int)
    parser_MGLAlign.add_argument('--walk_len',       default=10,         type=int)
    parser_MGLAlign.add_argument('--cross_alpha', default=5, type=float)
    parser_MGLAlign.add_argument('--switch_prob', default=0.3, type=float)
    parser_MGLAlign.add_argument('--walk_every', default=8, type=int,help='每隔8轮重新随机游走一次')

    #映射模型
    parser_MGLAlign.add_argument('--mapping_type', type=str, default='new',help='new or saved 映射模型是已经训练好还是新训练一个模型')
    parser_MGLAlign.add_argument('--mapping_model', type=str, default='mlp',help='mlp or linear or other 映射模型的类型')
    parser_MGLAlign.add_argument('--mapping_dim', type=int, default=128,help='映射模型的输出维度')
    parser_MGLAlign.add_argument('--mapping_epochs', type=int, default=150,help='映射模型的训练次数')

    parser_MGLAlign.add_argument('--noise_level', type=float, default=0.1)
    #后处理
    parser_MGLAlign.add_argument('--refine_type', type=str, default='refina',help='refina,gradalign或者为空')
    parser_MGLAlign.add_argument('--n_iter', type=int, default=20,help='后处理如果是refina的训练轮次')
    parser_MGLAlign.add_argument('--n-update', type=int, default=-1,
                        help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')
    parser_MGLAlign.add_argument('--token-match', type=float, default=-1,
                        help="Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")

    #超参数
    parser_MGLAlign.add_argument('--coe_consistency', type=float, default=0.8, help='balancing consistency and adaptivity loss')
    parser_MGLAlign.add_argument('--threshold', type=float, default=0.01, help='confidence threshold for adaptivity loss')
    parser_MGLAlign.add_argument('--embedding_type', type=str, default='new',help='new or saved 嵌入模型是已经训练好还是新训练一个模型')
    parser_MGLAlign.add_argument('--mask_rate', type=float, default=0.2,help='对锚点共现或者二阶邻接矩阵进行掩码的比例')
    parser_MGLAlign.add_argument("--top_rate", type=float, default=0.1,help='头尾节点占全部节点的比例')
    parser_MGLAlign.add_argument("--link_topk", type=int, default=10,help='稀疏处理，这里采用了阈值拦截，保留了具有前K个计算相似性的边')
    parser_MGLAlign.add_argument("--_lambda", type=float, default=0.02,help='控制元边生成器的损失和对比损失之间的比例')
    parser_MGLAlign.add_argument("--ssl_temp", type=float, default=0.3,help='是𝜏′是超参数，在softmax中称为温度')
    parser_MGLAlign.add_argument("--convergence", type=float, default=40,help='就是r，控制𝑃𝑜𝑝(𝑣𝑖)增加到1左右的速度的超参数')
    parser_MGLAlign.add_argument("--alpha", type=float, default=0.8,help='控制intra和inter损失的比例')
    parser_MGLAlign.add_argument("--beta", type=float, default=0.8,help='控制元学习损失和元测试损失的比例')
    parser_MGLAlign.add_argument("--gamma", type=float, default=0.8,help='控制有监督损失和无监督损失')
    parser_MGLAlign.add_argument('--weight0', type=float, default=1)
    parser_MGLAlign.add_argument('--weight1', type=float, default=0.8)
    parser_MGLAlign.add_argument('--weight2', type=float, default=0.5)
    parser_MGLAlign.add_argument('--weight3', type=float, default=0.5)
    parser_MGLAlign.add_argument('--weight4', type=float, default=0.5)
    parser_MGLAlign.add_argument('--weight5', type=float, default=0.5)
    parser_MGLAlign.add_argument('--weight6', type=float, default=0.5)

    #学习率和权重衰减
    parser_MGLAlign.add_argument("--local_lr", type=float, default=0.01,help="local_lr就是alpha,用来快速进行梯度下降的，对应公式seta'=seta-alpha*seta")
    parser_MGLAlign.add_argument("--lr", type=float, default=0.001)
    parser_MGLAlign.add_argument("--weight_decay", type=float, default=0.01)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    source_dataset = Dataset(args.source_dataset,'s')
    target_dataset = Dataset(args.target_dataset,'t')
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
    torch.cuda.set_device(args.device)
    algorithm = args.algorithm

    if algorithm == "IsoRank":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = IsoRank(source_dataset, target_dataset, args.H, args.alpha, args.max_iter, args.tol, train_dict=train_dict)
    elif algorithm == "FINAL":
        train_dict = None
        if args.train_dict != "":
            train_dict = graph_utils.load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        model = FINAL(source_dataset, target_dataset, H=args.H, alpha=args.alpha, maxiter=args.max_iter, tol=args.tol, train_dict=train_dict)
    elif algorithm == "REGAL":
        model = REGAL(source_dataset, target_dataset, max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=args.buckets,
                      gammastruc = args.gammastruc, gammaattr = args.gammaattr, normalize=True, num_top=args.num_top)
    elif algorithm == "IONE":
        model = IONE(source_dataset, target_dataset, gt_train=args.train_dict, epochs=args.epochs, dim=args.dim, seed=args.seed, learning_rate=args.lr)
    elif algorithm == "GAlign":
        model = GAlign(source_dataset, target_dataset, args)
    elif algorithm == "PALE":
        model = PALE(source_dataset, target_dataset, args)
    elif algorithm == "CENALP":
        model = CENALP(source_dataset, target_dataset, args)
    elif algorithm == "MGLAlign":
        model = MGLAlign(source_dataset, target_dataset, args)
    elif algorithm == "NAME":
        model = NAME(source_dataset, target_dataset, args)
    elif algorithm == "GradAlignPlus":
        model = GradAlignPlus(source_dataset, target_dataset, args)
    elif algorithm == "NextAlign":
        model = NextAlign(source_dataset, target_dataset, args)
    else:
        raise Exception("Unsupported algorithm")

    S = model.align()
    
    print("-"*100)
    if args.train_type == 'super': 
        #test_dict = graph_utils.load_gt(args.test_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        test_dict = graph_utils.load_gt(args.test_dict,format='dict')
        print('super test dict:'+str(len(test_dict)))
        acc, MAP, top5, top10, AUC,top30,Hit30 = get_statistics(S, test_dict, use_greedy_match=False, get_all_metric=True)
    elif args.train_type == 'unsuper':
        #groundtruth = graph_utils.load_gt(args.groundtruth, format='dict')
        print('unsuper groundtruth dict:'+str(len(groundtruth)))
        acc, MAP, top5, top10, AUC,top30,Hit30 = get_statistics(S, groundtruth, use_greedy_match=False, get_all_metric=True)
    print("MAP: {:.4f}".format(MAP))
    print("AUC: {:.4f}".format(AUC))
    print("Accuracy: {:.4f}".format(acc))
    #print("Precision_1: {:.4f}".format(top1))
    print("Precision_5: {:.4f}".format(top5))
    print("Precision_10: {:.4f}".format(top10))
    print("Precision_30: {:.4f}".format(top30))
    print("Hit-Precision_30: {:.4f}".format(Hit30))
    print("-"*100)
    print('Running time: {}'.format(time()-start_time))
    """ score, _ = refina_utils.score_alignment_matrix(S, topk = 1, true_alignments = true_align)
    mnc = refina_utils.score_MNC(alignment_matrix, adjA, adjB)
    print("Top 1 accuracy: %.5f" % score)
    print("MNC: %.5f" % mnc) """
    
    """ csv_filename = str(int(args.mixup_weight))+'_experiment_results.csv'
    with open(csv_filename, 'a+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([acc,MAP,AUC,top1,top5,top10]) """
