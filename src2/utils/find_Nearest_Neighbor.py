import torch
import torch.nn as nn
from numba import jit, int32
import numpy as np
from sklearn.neighbors import KDTree

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pdb
import util
import time


class find_NN(nn.Module):

    def forward(self,pcloud, P1, K):

        """args
            pcloud: point cloud with shape (1, num_points, 3)
            P1 ; query point with shape (3)



        """
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pcloud = torch.reshape(pcloud, (-1,3))
        P1 = torch.reshape(P1, (1,3))
        dist = torch.norm(pcloud-P1, dim=1, p=None)
        knn = dist.topk(K, largest=False)
        distances, indices = knn.values, knn.indices
        #max_distance = torch.max(distances)


        #return pcloud[indices].detach().cpu(), torch.reshape(indices,(1,-1)).detach().cpu()
        return pcloud[indices], torch.reshape(indices,(1,-1))
class find_NN_cuda(nn.Module):

    def forward(self,pcloud, P1, K):

        """args
            pcloud: point cloud with shape (1, num_points, 3)
            P1 ; query point with shape (3)



        """
        pcloud = torch.reshape(pcloud, (-1,3))
        P1 = torch.reshape(P1, (1,3))
        dist = torch.norm(pcloud-P1, dim=1, p=None)
        knn = dist.topk(K, largest=False)
        distances, indices = knn.values, knn.indices
        #max_distance = torch.max(distances)


        return pcloud[indices], torch.reshape(indices,(1,-1))


class find_NN_batch(nn.Module):        

    def forward(self,pcloud, P1, K):

        """args
            pcloud: point cloud with shape (batch_size, 3, num_points)
            P1 ; query point with shape (batch_size,3)

            output a : nearest point pcloud (batch_size, 3, new_npoints)
            output b : nearest point indices (batch_size, new_npoints)

        """
        pcloud = torch.transpose(pcloud,1,2)
        batch_size =  pcloud.shape[0]
        
       
        a = []
        b = []


        for i in range(batch_size):
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tmp_pcloud = torch.reshape(pcloud[i], (-1,3))
            tmp_P1 = torch.reshape(P1[i], (1,3))

            #util.interact(locals())
            #pdb.set_trace()
            dist = torch.norm(tmp_pcloud-tmp_P1, dim=1, p=None)
            knn = dist.topk(K, largest=False)
            distances, indices = knn.values, knn.indices    
            
            #tree = KDTree(tmp_pcloud.detach().cpu())
            #distances, indices = tree.query(tmp_P1.detach().cpu(), K)
            #max_distance = torch.max(distances)
            a.append(tmp_pcloud[indices].unsqueeze(0))
            b.append(indices.unsqueeze(0))
        b = torch.cat(b,0)
        b = torch.reshape(b, (batch_size, -1))
    
        return torch.transpose(torch.cat(a,0),1,2), b


if __name__ == '__main__':

    s = time.time()
    #pcloud = torch.Tensor([[0,1,2], [4,5,6], [3,5,2], [100,1,2]])
    pcloud = torch.rand(120,3000,3)
    P1 = torch.rand(120,3)
    K = 2000
    find_NN_batch1 = find_NN_batch()
    pc, indices = find_NN_batch1(pcloud, P1, K)
    t = time.time()
    print(t-s)











