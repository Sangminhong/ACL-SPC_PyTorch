import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from pytorch3d.ops.knn import knn_gather, knn_points

dtype = torch.double
device = torch.device("cuda")



def PolyConv(Input,indices,conv_p,conv_w):
    
    Input_adj = knn_gather(Input,indices)
    x = Input.unsqueeze(2).unsqueeze(-1) #Input_adj[:,0,:].unsqueeze(1).unsqueeze(-1)
    y = Input_adj[:,:,:].unsqueeze(-1)
    x_repeat = x*torch.ones_like(y)
    fxy = (torch.cat([torch.ones_like(y),x_repeat,y,torch.pow(x_repeat,2),x_repeat*y,torch.pow(y,2)],-1))*(conv_p.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    yfxy = fxy * y
    yfxy = torch.mean(torch.sum(yfxy,-1),dim=2)


    x_prime  = torch.cat([2*torch.ones_like(x),2*x,torch.zeros_like(x),2*torch.pow(x,2),torch.zeros_like(x),(2/3)*torch.ones_like(x)],-1)
    
    fx  = x_prime*(conv_p.unsqueeze(0).unsqueeze(0).unsqueeze(0))


    fx = torch.sum(fx,-1)[:,:,0]

    fxy_cond = torch.true_divide(yfxy,fx+0.0001)
    fxy_cond = conv_w(fxy_cond)
    return fxy_cond


def pool_max(Input, Adj, C, pool_num, conv_num,b):
    x = Input[:,Adj[b, :conv_num[1]]]
    x = scatter_max(x,C[b,:conv_num[1]],dim=1)[0]
    x = x[:, :pool_num[0]].float()
    return x

def pool_mean(Input, Adj, C, pool_num, conv_num,b):
    x = Input[:,Adj[b, :conv_num[1]]]
    x = scatter_mean(x,C[b,:conv_num[1]],dim=1)
    x = x[:, :pool_num[0]].float()
    return x


def network(Input,CONV, IN):
    Input = torch.transpose(Input,2,1)

    k = 6
    pool = True
    
    # First layer
    _,indices,_= knn_points(Input,Input,K=k)
    #x1 = torch.tanh(PolyConv(Input.float(),indices,CONV[0],CONV[1]))
    x1 = torch.transpose(torch.tanh(IN[0](torch.transpose(PolyConv(Input.float(),indices,CONV[0],CONV[1]),1,2))),1,2)
    pool_ind1 = torch.randint(x1.shape[1], (512,))
    x1 = torch.max(knn_gather(x1,indices[:,pool_ind1,:]),2)[0]


    
    # Second layer    
    _,indices,_= knn_points(Input[:,pool_ind1],Input[:,pool_ind1],K=k)
    #x2 = torch.tanh(PolyConv(x1,indices,CONV[2],CONV[3]))
    x2 = torch.transpose(torch.tanh(IN[1](torch.transpose(PolyConv(x1,indices,CONV[2],CONV[3]),1,2))),1,2)
    pool_ind2 = torch.randint(x2.shape[1], (128,))
    x2 = torch.max(knn_gather(x2,indices[:,pool_ind2,:]),2)[0]

                
    # Third layer
    _,indices,_= knn_points(Input[:,pool_ind1[pool_ind2]],Input[:,pool_ind1[pool_ind2]],K=k)
    #x3 = torch.tanh(PolyConv(x2,indices,CONV[4],CONV[5]))
    x3 = torch.transpose(torch.tanh(IN[2](torch.transpose(PolyConv(x2,indices,CONV[4],CONV[5]),1,2))),1,2)
    pool_ind3 = torch.randint(x3.shape[1], (32,))
    x3 = torch.max(knn_gather(x3,indices[:,pool_ind3,:]),2)[0]

        
    # Forth layer
    _,indices,_= knn_points(Input[:,pool_ind1[pool_ind2[pool_ind3]]],Input[:,pool_ind1[pool_ind2[pool_ind3]]],K=k)
    #x4 = torch.tanh(PolyConv(x3,indices,CONV[6],CONV[7]))
    x4 = torch.transpose(torch.tanh(IN[3](torch.transpose(PolyConv(x3,indices,CONV[6],CONV[7]),1,2))),1,2)

    Output = torch.mean(x4,1)
    return Output
