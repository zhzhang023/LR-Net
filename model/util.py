import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
import math
import model.Config as cfg

'''
#### ----Random Sampling and Grouping---- ####
'''
# Sampling
def random_sampling(pc, num_sample, replace=None, return_choices=False,weight=None):
    """
    Input: N x C,
    output: num_sample x C
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    if weight is not None:
        weight=weight.detach().cpu().numpy()
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace,p=weight)
    if return_choices:
        return pc[choices], torch.from_numpy(choices)
    else:
        return pc[choices]

def random_sampling_batch(pc, num_sample, return_choices=False):
    """
    Input: pc [B, N, C]
    Return: [B, num_sample, C]
    """
    PC_out=[]
    choices_out=[]
    B,N,C=pc.shape
    for i in range(B):
        pc_temp,choices_temp=random_sampling(pc[i,...],num_sample,return_choices=True)
        PC_out.append(pc_temp.unsqueeze(0))
        choices_out.append(choices_temp.unsqueeze(0))
    PC_out=torch.cat(PC_out,dim=0)
    choices_out=torch.cat(choices_out,dim=0)
    # PC_out=PC_out.reshape(B,num_sample,C)
    # choices_out=choices_out.reshape(B,-1)
    if return_choices:
        return PC_out,choices_out
    else:
        return PC_out

# Indexing
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape).cuda()
    new_points = points[batch_indices, idx.cuda(), :]
    return new_points

def torch_gather_nd(points,idx):
    '''
    Input:
        points: [B,N,C]
        idx:[B,nsample,K,2]  0:batch_idx 1:point_idx
    Return:
            [B,nsample,K,3]
    '''
    return points[idx[:,:,:,0],idx[:,:,:,1],:]

def index_points_feature(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S,K]
    Return:
        new_points:, indexed points data, [B, S, K,C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long,device=points.device).view(view_shape).repeat(repeat_shape)
    idns_stack=torch.stack([batch_indices,idx.cuda()],dim=3)
    new_points = torch_gather_nd(points,idns_stack)
    return new_points

# Grouping
def kdtree_based_knn(queries,points,k):
    B,N,_=points.shape
    out_idx = []
    for i in range(B):
        pts=points[i,...]
        qts=queries[i,...]
        tree = cKDTree(pts.detach().cpu().numpy())
        _, indice = tree.query(qts.detach().cpu().numpy(), k)
        choices = np.random.choice(indice.shape[1], k, replace=False)
        indice=indice[:,choices]
        out_idx.append(torch.from_numpy(indice).unsqueeze(0))
    out_idx=torch.cat(out_idx,dim=0).to(points.device)
    return out_idx

def kdtree_based_ball_query(queries,points,radius,k):
    B, N, _ = points.shape
    _,S,_=queries.shape
    out_idx = []
    for i in range(B):
        pts = points[i, ...]
        qts = queries[i, ...]
        tree = cKDTree(pts.detach().cpu().numpy())
        indice=tree.query_ball_point(qts.detach().cpu().numpy(),radius)
        index_temp=[]
        for j in range(S):
            if len(indice[j])<10:
                _,idx_knn=tree.query(qts[j].detach().cpu(),k)
                index_temp.append(torch.from_numpy(idx_knn).unsqueeze(0))
                continue
            choices = np.random.choice(len(indice[j]), k, replace=True)
            tt = torch.from_numpy(np.array(indice[j])[choices])
            index_temp.append(tt.unsqueeze(0))
        index_temp=torch.cat(index_temp,dim=0)
        out_idx.append(index_temp.unsqueeze(0))
    out_idx = torch.cat(out_idx, dim=0).to(points.device)
    return out_idx

def Angel_calculator(v1,v2):
    '''
    :param v1: [B,N,K,3]
    :param v2: [B,N,K,3]
    :return: [B,N,K,1]
    '''

    M=torch.sqrt(torch.sum(v1*v1,dim=-1))
    N=torch.sqrt(torch.sum(v2*v2,dim=-1))
    cos_theta=torch.sum(v1*v2,dim=-1)/((M*N)+1e-4)
    theta=torch.acos(cos_theta)
    out=torch.rad2deg_(theta)
    return out

def TIF_forknn(grouped_xyz, sample_point,local_center):
    """
    :param grouped_xyz: [B,N,K,3]
    :param sample_point: [B,N,3]
    :param local_center: [B,N,3]
    :return:
    """
    # 获取一些基本参数
    B,N,K,C=grouped_xyz.shape
    M = local_center.view(B, N, 1, C).repeat(1, 1, K, 1)
    G = grouped_xyz
    S = sample_point.view(B, N, 1, C).repeat(1, 1, K, 1)
    Z = torch.zeros_like(grouped_xyz)
    Z[...,-1]= 1
    # 构造基本向量
    # Grouped_xyz---local center(Mean)
    G2M=M-G
    M2G=G-M
    # Sample Point --- local center(Mean)
    S2M=M-S
    M2S=S-M
    # Sample Point --- Grouped xyz
    S2G=G-S
    G2S=S-G
    # 计算角度
    A_MGS=Angel_calculator(G2M,G2S).view(B, N, K, 1)
    A_MSG=Angel_calculator(S2M,S2G).view(B, N, K, 1)
    A_SMG=Angel_calculator(M2S,M2G).view(B, N, K, 1)
    A_GSZ=Angel_calculator(S2G,Z).view(B, N, K, 1)

    # 计算边长
    L_GM = torch.sqrt(torch.sum(G2M*G2M, dim=-1)).view(B, N, K, 1)
    L_GS = torch.sqrt(torch.sum(G2S * G2S, dim=-1)).view(B, N, K, 1)
    # 计算局部区域特征
    L_SM=torch.sqrt(torch.sum(S2M * S2M, dim=-1)).view(B, N, K, 1)

    TIF = torch.cat((A_MGS, A_MSG, A_SMG, A_GSZ, L_GM, L_GS, L_SM), dim=-1)
    loc=torch.where(torch.isnan(TIF))
    TIF[loc]=1e-4
    return TIF

def sample_and_group(npoint,radius,nsample, xyz, feature, idx=None,returnfps=False,knn=True,dialated_ratio=2,RIF=True):

    B, N, C = xyz.shape
    if idx is None:
        new_xyz,sample_idx=random_sampling_batch(xyz,npoint,return_choices=True)
    else:
        sample_idx=idx[:,0:npoint]
        new_xyz=index_points(xyz,sample_idx)
    if knn:
        idx=kdtree_based_knn(new_xyz,xyz,nsample)
    else:
        idx = kdtree_based_ball_query(new_xyz, xyz, radius,nsample)
    grouped_xyz = index_points_feature(xyz, idx) # [B, npoint, nsample, C]
    if feature is not None:
        _, _, D = feature.shape
        grouped_feature=index_points_feature(feature,idx) #[B,npoint,nsample,D]
        if RIF:
            center_xyz = (torch.sum(grouped_xyz, dim=-2)) / nsample
            TIF_feature = TIF_forknn(grouped_xyz, new_xyz, center_xyz) # test global center
            new_feature=torch.cat([TIF_feature,grouped_feature],dim=-1) #[B,npoint,nsample,C+D]
        else:
            new_feature = torch.cat([grouped_xyz, grouped_feature], dim=-1)
    else:
        # Transformation-Invariant feature
        center_xyz = (torch.sum(grouped_xyz, dim=-2))/nsample
        if RIF:
            new_feature=TIF_forknn(grouped_xyz,new_xyz,center_xyz)
        else:
            new_feature=grouped_xyz
    if returnfps:
        return new_xyz,new_feature,sample_idx
    else:
        return new_xyz, new_feature

'''
#### ----Encoding Modules ---- ####
'''
class ResidualMLP(nn.Module):
    def __init__(self,channel,mode='2d'):
        super(ResidualMLP, self).__init__()
        if mode=='2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif mode=='1d':
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        else:
            raise NotImplementedError
        self.mode=mode
        self.net1=nn.Sequential(
            conv(in_channels=channel,out_channels=channel,kernel_size=1),
            bn(channel),
            nn.ReLU(),
        )
        self.net2=nn.Sequential(
            conv(in_channels=channel,out_channels=channel,kernel_size=1),
            bn(channel),
        )
        self.act=nn.ReLU()
    def forward(self,inputs):
        if self.mode=='2d':
            inputs=inputs.permute(0,3,2,1)
            outputs=self.act(self.net2(self.net1(inputs)+inputs))
            outputs=outputs.permute(0,3,2,1)
        else:
            inputs = inputs.permute(0, 2, 1)
            outputs = self.act(self.net2(self.net1(inputs) + inputs))
            outputs = outputs.permute(0, 2, 1)
        return outputs

class SharedMLP(nn.Module):
    def __init__(self,in_channels,out_channels,with_bn=False,mode='2d'):
        super().__init__()
        if mode=='2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif mode=='1d':
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        else:
            raise NotImplementedError
        self.mode=mode
        if not isinstance(out_channels,(list,tuple)):
            out_channels=[out_channels]
        layers = []
        for oc in out_channels:
            if with_bn:
                layers.extend([
                    conv(in_channels, oc, 1),
                    bn(oc),
                    nn.ReLU(),
                ])
            else:
                layers.extend([
                    conv(in_channels, oc, 1),
                    nn.ReLU(),
                ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.mode=='2d':
            inputs=inputs.permute(0,3,2,1)
            return self.layers(inputs).permute(0,3,2,1)
        else:
            inputs = inputs.permute(0, 2, 1)
            return self.layers(inputs).permute(0, 2, 1)

### Feature Propagation ###
def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

''' LSA Encoder '''
class LSA_feature_generator(nn.Module):
    def __init__(self,in_channel,out_channel,point_channel=64,local_channel=64,with_bn=True):
        super().__init__()
        conv = nn.Conv2d
        self.point_encoding_layer = conv(in_channel, point_channel, 1)
        self.local_encoding_layer = conv(in_channel, local_channel, 1)
        self.final_encoding_layer = SharedMLP(in_channels=point_channel+local_channel,
                                              out_channels=out_channel,with_bn=with_bn)
    def forward(self, input):
        input=input.permute(0,3,2,1)
        B,C,K,N=input.shape
        point_feature=self.point_encoding_layer(input)
        local_feature=self.local_encoding_layer(input)
        local_feature=F.avg_pool2d(local_feature,(local_feature.size(2),1)).repeat(1,1,K,1)
        out_feature=self.final_encoding_layer(torch.cat([point_feature,local_feature],dim=1).permute(0,3,2,1))
        return out_feature

''' Grouped Transformer '''
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_sparse_avg = torch.mean(x,dim=0)
        # Apply 1D convolution along the channel dimension
        y_avg = self.conv_avg(y_sparse_avg.unsqueeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).squeeze(-1)
        y=y_avg
        # y is (batch_size, channels) tensor

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor
        # braodcast multiplication
        x=x*y
        return x

class G_Transformer(nn.Module):
    def __init__(self, channels, gp):
        super().__init__()
        mid_channels = channels
        self.gp = gp
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x=x.permute(0,2,1).contiguous()
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch // self.gp, nums)
        x_q = x_q.permute(0, 1, 3, 2)  # B x gp x num x C'

        x_k = self.k_conv(x)  # B x C x N
        x_k = x_k.reshape(bs, self.gp, ch // self.gp, nums)  # B x gp x C' x nums

        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)  # B x gp x N x N
        energy = torch.sum(energy, dim=1, keepdims=False)

        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x_r = torch.matmul(x_v, attn)
        x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x = x + x_r
        return x.permute(0,2,1).contiguous()

## GeM
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self,x):
        batch_size=x.shape[0]
        out=[]
        for i in range(batch_size):
            temp=x[i,...]
            temp=self._pool(temp)
            out.append(temp)
        out=torch.stack(out)
        return out

    def _pool(self, x):
        '''
        Input:
            x: [B,N,C]
        return:
            global_feature: [B,C]
        '''
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p).permute(0,2,1)
        temp = torch.adaptive_avg_pool1d(temp,1).squeeze()          # Apply ME.MinkowskiGlobalAvgPooling
        return temp.pow(1./self.p)

class GeM_pool(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_pool, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self,x):
        '''
        Input:
            x: [B,N,C]
        return:
            global_feature: [B,C]
        '''
        # This implicitly applies ReLU on x (clamps negative values)
        temp = x.clamp(min=self.eps).pow(self.p).permute(0,2,1)
        temp = torch.adaptive_avg_pool1d(temp,1).squeeze()          # Apply ME.MinkowskiGlobalAvgPooling
        return temp.pow(1./self.p)
