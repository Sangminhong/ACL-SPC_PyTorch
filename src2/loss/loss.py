import torch
import torch.nn as nn
import torch.nn.functional as f
import sys
sys.path.append('../')
import numpy as np
from knn_cuda import KNN
from pykdtree.kdtree import KDTree
import open3d as o3d
from pytorch3d.ops.knn import knn_gather, knn_points
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()

        
        
def PolyConv_loss(x,Input,indices,conv_p,conv_w):
    Input_adj = knn_gather(Input,indices)
    x = x.unsqueeze(2).unsqueeze(-1) #Input_adj[:,0,:].unsqueeze(1).unsqueeze(-1)
    y = Input_adj[:,:,:].unsqueeze(-1)
    x_repeat = x*torch.ones_like(y)
    fxy = (torch.cat([torch.ones_like(y),x_repeat,y,torch.pow(x_repeat,2),x_repeat*y,torch.pow(y,2)],-1))*(conv_p.detach().unsqueeze(0).unsqueeze(0).unsqueeze(0))
    yfxy = fxy * y
    yfxy = torch.mean(torch.sum(yfxy,-1),dim=2)
    x_prime  = torch.cat([2*torch.ones_like(x),2*x,torch.zeros_like(x),2*torch.pow(x,2),torch.zeros_like(x),(2/3)*torch.ones_like(x)],-1)
    fx  = x_prime*(conv_p.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    fx = torch.sum(fx,-1)[:,:,0]
    fxy_cond = torch.true_divide(yfxy,fx+0.0001)
    fxy_cond = F.linear(fxy_cond, conv_w.weight.detach(), conv_w.bias.detach())
    return fxy_cond



def estimate_normals(points,k):
    batch_size = points.size(0)
    points_mean = points.mean(1,keepdim=True) 
    points_centered = points - points_mean
    knn_indices, _ = get_nearest_neighbors_indices_batch(points_centered.detach().cpu().numpy(),points_centered.detach().cpu().numpy(),k)
    knn_indices = torch.LongTensor(knn_indices).to(points.device)
    knn_points = []
    for b in range(batch_size):   
        knn_points.append(points[b,knn_indices[b]].unsqueeze(0))
    knn_points = torch.cat(knn_points,0)
    pt_mean = knn_points.mean(2, keepdim=True)
    central_diff = knn_points - pt_mean
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    covariances = per_pt_cov.mean(2)    
    print(covariances.device)
    curvatures, local_coord_frames = torch.symeig(covariances, eigenvectors=True)    
    print(curvatures.shape)
    normals = local_coord_frames[:, :, :, 0]
    print(normals.shape)
    return normals, knn_indices




def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.
    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances



def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.
    Args:
        points1 (batch, 3, num_on_points)
        points2 (batch, 3, num_on_points)
    '''
    points1 = torch.transpose(points1,1,2)
    points2 = torch.transpose(points2,1,2)
    batch_size, T1, _ = points1.size()
    _, T2, _ = points2.size()

    points1 = points1.view(batch_size, T1, 1, 3)
    points2 = points2.view(batch_size, 1, T2, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)
    return chamfer1, chamfer2


def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """
    # UHD from MPC: https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/completeness.py
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """


    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist

def distChamfer(a, b):
    """
    :param a: Pointclouds Batch x nul_points x dim
    :param b:  Pointclouds Batch x nul_points x dim
    :return:
    -closest point on b of points from a
    -closest point on a of points from b
    -idx of closest point on b of points from a
    -idx of closest point on a of points from b
    Works for pointcloud of any dimension
    """
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()


def chamfer_distance_kdtree(x, y):
    ''' KD-tree based implementation of the Chamfer distance.
    Args:
        points1 (batch, 3, num_on_points)
        points2 (batch, 3, num_on_points)
    '''

    x = torch.transpose(x,1,2)
    y = torch.transpose(y,1,2)


    x_nn = knn_points(x, y, K=1)
    y_nn = knn_points(y, x, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # Apply point reduction
    cham_x = torch.sqrt(cham_x)  # (N,)
    cham_y = torch.sqrt(cham_y)  # (N,)

    return cham_x.mean(1), cham_y.mean(1)


def normal_loss(points,k=6, epsilon=0.001):
    points = torch.transpose(points,1,2)
    batch_size = points.size(0)
    points_mean = points.mean(1,keepdim=True) 
    points_centered = points - points_mean
    knn_indices, _ = get_nearest_neighbors_indices_batch(points_centered.detach().cpu().numpy(),points_centered.detach().cpu().numpy(),k)
    knn_indices = torch.LongTensor(knn_indices).to(points.device)
    knn_points = []
    for b in range(batch_size):   
        knn_points.append(points[b,knn_indices[b]].unsqueeze(0))
    
    knn_points = torch.cat(knn_points,0)
    central_diff = knn_points
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    covariances = per_pt_cov.mean(2)    
    curvatures, local_coord_frames = torch.linalg.eigh(covariances) 
    vertex_normal = local_coord_frames[:, :, :, 0]

    knn_normals = []
    for b in range(batch_size):   
        knn_normals.append(vertex_normal[b,knn_indices[b]].unsqueeze(0))
    
    knn_normals = torch.cat(knn_normals,0)
    loss = l1_loss(knn_normals[:,:,1:,:]-knn_normals[:,:,0,:].unsqueeze(2),torch.zeros_like(knn_normals[:,:,1:,:]))
    return loss
    
    
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        
        return self.loss(input, target)



class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.loss = ChamferDistance()       
        self.device = device

    def forward(self, xyz1, xyz2):
        return self.loss(xyz1, xyz2)



