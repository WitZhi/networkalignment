import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn import init
import json
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform

def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def get_act_function(activate_function):
    """
    Get activation function by name
    :param activation_fuction: Name of activation function 
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function


class CombineModel(nn.Module):
    def __init__(self):
        super(CombineModel, self).__init__()
        self.thetas = nn.Parameter(torch.ones(3))

    
    def loss(self, S1, S2, S3, id2idx_augment):
        S = self.forward(S1, S2, S3)
        S_temp = torch.zeros(S.shape)
        for k,v in id2idx_augment.items():
            S_temp[int(k),v] = 1
        
        S = S / torch.sqrt((S**2).sum(dim=1)).view(S.shape[0],1)
        loss = -(S * S_temp).mean()
        return loss


    def forward(self, S1, S2, S3):
        theta_sum = torch.abs(self.thetas[0]) + torch.abs(self.thetas[1]) + torch.abs(self.thetas[2])
        return (torch.abs(self.thetas[0])/theta_sum) * S1 + (torch.abs(self.thetas[1])/theta_sum) * S2 + (torch.abs(self.thetas[2])/theta_sum) * S3


class Combine2Model(nn.Module):
    def __init__(self):
        super(Combine2Model, self).__init__()
        self.thetas = nn.Parameter(torch.ones(2))


    def loss(self, S1, S2, id2idx_augment):
        S = self.forward(S1, S2)
        S_temp = torch.zeros(S.shape)
        for k,v in id2idx_augment.items():
            S_temp[int(k),v] = 1
        
        S = S / torch.max(S, dim=1)[0].view(S.shape[0],1)
        loss = -(S * S_temp).mean()
        # loss = (S - 3 * torch.eye(len(S))).mean()
        return loss

    def forward(self, S1, S2):
        return torch.abs(self.thetas[0]) * S1 + torch.abs(self.thetas[1]) * S2

#来自mixup的图卷积层
class GraphConv(MessagePassing):
    def __init__(self, activate_function, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, A_hat, x_cen):
        #以下两行融合了消息传递和聚合
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(A_hat, size=None, h=h)
        #消息传递结果与经过线性变换后的节点中心特征相加，也就是消息传递以及聚合以后的更新操作
        aggr_out = aggr_out + self.lin(x_cen)
        if self.activate_function is not None:
            aggr_out = self.activate_function(aggr_out)
        return aggr_out

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(nn.Module):
    """
    The GCN multistates block
    """
    def __init__(self, activate_function, input_dim, output_dim):
        """
        activate_function: Tanh
        input_dim: input features dimensions
        output_dim: output features dimensions
        """
        super(GCN, self).__init__()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        init_weight(self.modules(), activate_function)
    
    def forward(self, input, A_hat,x_cen = None):
        output = self.linear(input)
        output = torch.matmul(A_hat, output)
        #仿照MixupForGraph
        """ print(self.linear(x_cen).shape)
        print(self.input_dim, self.output_dim) """
        #能提示4%左右的性能
        if x_cen is not None:
            output = output + self.linear(x_cen)

        if self.activate_function is not None:
            output = self.activate_function(output)
        return output


class G_Align(nn.Module):
    """
    Training a multilayer GCN model
    """
    def __init__(self, activate_function, num_GCN_blocks, input_dim, output_dim, \
                num_source_nodes, num_target_nodes,source_ids=None,target_ids=None,shuffled_source_A_hat=None,shuffled_target_A_hat=None, source_feats=None, target_feats=None,mixup=False):
        """
        :params activation_fuction: Name of activation function
        :params num_GCN_blocks: Number of GCN layers of model
        :params input_dim: The number of dimensions of input
        :params output_dim: The number of dimensions of output
        :params num_source_nodes: Number of nodes in source graph
        :params num_target_nodes: Number of nodes in target graph
        :params source_feats: Source Initialized Features
        :params target_feats: Target Initialized Features
        """
        super(G_Align, self).__init__()
        self.num_GCN_blocks = num_GCN_blocks 
        self.source_feats = source_feats
        self.target_feats = target_feats
        input_dim = self.source_feats.shape[1]
        self.input_dim = input_dim
        if mixup:
            with open('best_param.json','r') as file:
                data = json.load(file)
            self.lam = data[0]['lam']

            self.shuffled_source_A_hat = shuffled_source_A_hat
            self.shuffled_target_A_hat = shuffled_target_A_hat
            self.source_ids = source_ids
            self.target_ids = target_ids
            self.lam = np.random.beta(4.0, 4.0)

        # GCN blocks (emb)
        self.GCNs = []
        self.GraphConvs=[]
        for i in range(num_GCN_blocks):
            self.GraphConvs.append(GraphConv(activate_function,input_dim, output_dim))
            self.GCNs.append(GCN(activate_function, input_dim, output_dim))
            input_dim = self.GCNs[-1].output_dim
        self.GCNs = nn.ModuleList(self.GCNs)
        self.GraphConvs = nn.ModuleList(self.GraphConvs)
        init_weight(self.modules(), activate_function)

    def forward(self, A_hat, net='s', mixup=False, new_feats=None):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix 
        :params net: Whether forwarding graph is source or target graph
        """
        if new_feats is not None:
            input = new_feats
        elif net == 's':
            input = self.source_feats
        else:
            input = self.target_feats
        emb_input = input.clone()
        outputs = [emb_input]
        if mixup:
            #mixup每一层
            if net == 's':
                old2new = self.source_ids
                shuffle_A_hat= self.shuffled_source_A_hat
            else:
                old2new = self.target_ids
                shuffle_A_hat = self.shuffled_target_A_hat
            shuffled_input = input[old2new]
            emb_input_mixup = input*self.lam + shuffled_input*(1-self.lam)
            outputs_mixup = [emb_input_mixup]
            #outputs_mixup = [emb_input]
            A_hat = A_hat*self.lam + shuffle_A_hat*(1-self.lam)
            
            #效果差了10%
            """ for i in range(self.num_GCN_blocks):
                GraphConv_output_i = self.GraphConvs[i](emb_input, A_hat.coalesce().indices(),emb_input)
                #GraphConv_output_i = F.dropout(GraphConv_output_i, p=0.4, training=self.training)
                GraphConv_output_i_shuffle = GraphConv_output_i[old2new]

                
                new_GraphConv_output_i = self.GraphConvs[i](emb_input, A_hat.coalesce().indices(),emb_input_mixup)
                new_GraphConv_output_i_shuffle = self.GraphConvs[i](shuffled_input, shuffle_A_hat.coalesce().indices(),emb_input_mixup)
                
                GraphConv_output_i_mixup = new_GraphConv_output_i*self.lam + new_GraphConv_output_i_shuffle*(1-self.lam)
                #GraphConv_output_i_mixup = F.dropout(GraphConv_output_i_mixup, p=0.4, training=self.training)

                outputs_mixup.append(GraphConv_output_i_mixup)

                emb_input = GraphConv_output_i
                shuffled_input = GraphConv_output_i_shuffle
                emb_input_mixup = GraphConv_output_i_mixup """

            #不用双分支，也不加线性层(降低总体性能)
            for i in range(self.num_GCN_blocks):
                GCN_output_i_mixup = self.GCNs[i](emb_input_mixup, A_hat)
                if 0:
                    GCN_output_i_shuffle = GCN_output_i_mixup[old2new]
                    GCN_output_i_mixup = self.lam*GCN_output_i_mixup+(1-self.lam)*GCN_output_i_shuffle
                outputs_mixup.append(GCN_output_i_mixup)
                emb_input_mixup=GCN_output_i_mixup
            
            #双分支
            """ for i in range(self.num_GCN_blocks):
                GCN_output_i = self.GCNs[i](emb_input, A_hat)
                GCN_output_i_shuffle = GCN_output_i[old2new]
                #GCN_output_i_shuffle = self.GCNs[i](shuffled_input, shuffle_A_hat,shuffled_input)
                new_GCN_output_i = self.GCNs[i](emb_input, A_hat,emb_input_mixup)
                new_GCN_output_i_shuffle = self.GCNs[i](shuffled_input, shuffle_A_hat,emb_input_mixup)
                GCN_output_i_mixup = new_GCN_output_i*self.lam + new_GCN_output_i_shuffle*(1-self.lam)
                outputs_mixup.append(GCN_output_i_mixup)
                emb_input = GCN_output_i
                shuffled_input = GCN_output_i_shuffle
                emb_input_mixup = GCN_output_i_mixup """
            """ for i in range(self.num_GCN_blocks):
                GCN_output_i = self.GCNs[i](emb_input, A_hat,emb_input)
                outputs_mixup.append(GCN_output_i)
                emb_input = GCN_output_i """
            return outputs_mixup
        else:
            for i in range(self.num_GCN_blocks):
                GCN_output_i = self.GCNs[i](emb_input, A_hat)
                outputs.append(GCN_output_i)
                emb_input = GCN_output_i
            return outputs



class StableFactor(nn.Module):
    """
    Stable factor following each node
    """
    def __init__(self, num_source_nodes, num_target_nodes, cuda=True):
        """
        :param num_source_nodes: Number of nodes in source graph
        :param num_target_nodes: Number of nodes in target graph
        """
        super(StableFactor, self).__init__()
        # self.alpha_source_trainable = nn.Parameter(torch.ones(num_source_nodes))
        self.alpha_source = torch.ones(num_source_nodes)
        self.alpha_target = torch.ones(num_target_nodes)
        self.score_max = 0
        self.alpha_source_max = None
        self.alpha_target_max = None
        if cuda:
            self.alpha_source = self.alpha_source.cuda()
            self.alpha_target = self.alpha_target.cuda()
        self.use_cuda = cuda
    
        
    def forward(self, A_hat, net='s'):
        """
        Do the forward 
        :param A_hat is the Normalized Laplacian Matrix
        :net: whether graph considering is source or target graph.
        """
        if net=='s':
            self.alpha = self.alpha_source
        else:
            self.alpha = self.alpha_target
        A_hat = A_hat.to_dense()
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat_new = (alpha_colum * (A_hat * alpha_colum).t()).t()
        return A_hat_new 


