import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import warnings
warnings.filterwarnings('ignore')

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
    
    def forward(self, input, A_hat):
        output = self.linear(input)
        output = torch.matmul(A_hat, output)
        if self.activate_function is not None:
            output = self.activate_function(output)
        return output


class G_Align(nn.Module):
    """
    Training a multilayer GCN model
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True,source_feats=None, target_feats=None,):
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
        self.num_GCN_blocks = num_layers 
        self.source_feats = source_feats
        self.target_feats = target_feats
        in_channels = self.source_feats.shape[-1]
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel, use_graph=use_graph, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, A_hat, net='s', new_feats=None):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix 
        :params net: Whether forwarding graph is source or target graph
        """
        layer_ = []
        
        if new_feats is not None:
            input = new_feats
        elif net == 's':
            input = self.source_feats
        else:
            input = self.target_feats
        emb_input = input.clone()
        outputs = [emb_input]
        x = emb_input

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        if A_hat.is_sparse:
                edge_index = A_hat.coalesce().indices()
                edge_weight = A_hat.coalesce().values()
        else:
            edge_index = A_hat.to_sparse().coalesce().indices()
            edge_weight = A_hat.to_sparse().coalesce().values()

        # 将edge_index和edge_weight向量转换为PyTorch张量
        edge_index = torch.tensor(edge_index)
        edge_weight = torch.tensor(edge_weight)
        
        for i, conv in enumerate(self.convs):
            # graph convolution with DIFFormer layer
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1-self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
            outputs.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)
        # outputs.append(x_out)
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
        alpha_colum = self.alpha.reshape(len(self.alpha), 1)
        if self.use_cuda:
            alpha_colum = alpha_colum.cuda()
        A_hat_new = (alpha_colum * (A_hat * alpha_colum).t()).t()
        return A_hat_new 

def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    N表示batch_size,即输入数据中样本的数量;
    H表示注意力头的数量,即多头注意力机制中使用的注意力头的数量;
    D表示每个样本的特征维度,即每个样本在注意力聚合过程中被映射到的新的特征维度.
    
    return output [N, H, D]
    '''
    if kernel == 'simple':
        # normalize input 归一化
        qs = qs / torch.norm(qs, p=2) # [N, H, M]
        ks = ks / torch.norm(ks, p=2) # [L, H, M]
        N = qs.shape[0]

        # numerator
        #计算乘积，得到一个大小为(H,M,D)的三维张量
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        #计算乘积，表示每个样本在每个注意力头上的注意力分数加权后的值
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]

        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        #计算值张量vs在第0维上的和
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
        #使用unsqueeze函数将vs_sum的第一维扩展为batch_size，然后使用repeat函数将其在第一维上复制batch_size次
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

        # denominator
        #创建一个形状为（L）的张量all_ones
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        #计算键张量ks在第0维上的和，得到一个形状为（H，M）的张量，表示每个注意力头上键张量的和
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        #计算查询张量qs和ks_sum的乘积
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        #将注意力分母张量attention_normalizer的最后一维扩展为1，为了将分母张量与注意力分子张量attention_num的形状对齐
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        #表示每个样本在每个注意力头上的注意力分母的值加上batch_size=N。这是为了避免分母为0的情况
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        #表示每个样本在每个注意力头上的注意力加权和
        attn_output = attention_num / attention_normalizer # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            #表示每个样本在每个注意力头上的注意力分数
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer.unsqueeze(2) # [N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        #[N,H,D]
        qs = qs / torch.norm(qs, p=2) # [N, H, M]
        ks = ks / torch.norm(ks, p=2) # [L, H, M]
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]
        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]
    
        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

def gcn_conv(x, edge_index, edge_weight):
    #从节点特征张量x中获取节点数量N和每个节点的特征数量H
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    #计算输入节点特征张量的归一化因子d_norm_in和d_norm_out，用于将邻居节点的特征进行归一化
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    #根据边权重和归一化因子计算边的权重value，并使用SparseTensor函数创建一个稀疏张量adj，表示图的邻接矩阵
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    #遍历每个节点特征的维度，使用matmul函数将邻接矩阵adj和节点特征张量x的对应维度进行矩阵乘法，得到一个形状为（N，D）的张量，表示每个节点的新特征
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    #将每个节点特征的结果沿着第二个维度进行堆叠，得到一个形状为（N，H，D）的张量，表示整个GCN卷积层的输出结果
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

def gcn_conv_attention(x, attention,edge_index, edge_weight):
    #从节点特征张量x中获取节点数量N和每个节点的特征数量H
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    attention = torch.sum(attention.mean(dim=1),dim=1)*200
    #normalized_attention = F.normalize(attention, p=2, dim=0).float()
    d = degree(col, N).float()
    d = torch.mul(d,attention.float())
    #计算输入节点特征张量的归一化因子d_norm_in和d_norm_out，用于将邻居节点的特征进行归一化
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    
    #根据边权重和归一化因子计算边的权重value，并使用SparseTensor函数创建一个稀疏张量adj，表示图的邻接矩阵
    #d_norm_in表示每个邻居节点特征在输入节点上的归一化因子，它是根据输入节点的度数进行计算的。
    #d_norm_out表示每个邻居节点特征在邻居节点上的归一化因子，它是根据邻居节点的度数进行计算的。
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    #遍历每个节点特征的维度，使用matmul函数将邻接矩阵adj和节点特征张量x的对应维度进行矩阵乘法，得到一个形状为（N，D）的张量，表示每个节点的新特征
    for i in range(x.shape[1]):
        gcn_conv_output.append( matmul(adj, x[:, i]) )  # [N, D]
    #将每个节点特征的结果沿着第二个维度进行堆叠，得到一个形状为（N，H，D）的张量，表示整个GCN卷积层的输出结果
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1) # [N, H, D]
    return gcn_conv_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple',
               use_graph=True,
               use_weight=True):
        super(DIFFormerConv, self).__init__()
        hidden_channels=300
        self.activation = F.relu
        self.GCN = GCNConv(in_channels,hidden_channels)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        #变换后的输出是一个三维张量，大小为（batch_size，num_heads，out_channels），表示每个样本的每个注意力头的输出特征[N,H,D]
    
        """ query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels) """
        """ if self.use_graph:
            query = self.GCN(query_input, edge_index)
            query = self.activation(query)
            key = self.GCN(source_input, edge_index)
            key = self.activation(key)
            #value = self.GCN(source_input, edge_index)
            value = gcn_conv(source_input.reshape(-1, 1, self.out_channels),edge_index,edge_weight)
            value = self.activation(value) """

        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        """ if self.use_graph:
            print(edge_weight)
            query = gcn_conv(query, edge_index, edge_weight)
            key = gcn_conv(key, edge_index, edge_weight)
            value = gcn_conv(value, edge_index, edge_weight) """

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query,key,value,self.kernel) # [N, H, D]
        # use input graph for gcn conv 如果使用图卷积
        if self.use_graph:
            final_output = gcn_conv_attention(value,attention_output, edge_index, edge_weight)
            #final_output = gcn_conv(value,edge_index, edge_weight)
            #final_output = attention_output+gcn_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output
        final_output = final_output.mean(dim=1)
        #final_output = F.normalize(final_output,p=2,dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output
