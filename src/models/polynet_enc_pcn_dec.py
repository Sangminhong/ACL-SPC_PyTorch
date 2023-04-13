if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.base_model import BaseModel
# import torchvision.models as models
from models.polyconv import network, PolyConv
from torch.autograd import Variable
import numpy as np
device = torch.device("cuda")
from pytorch3d.ops.knn import knn_points, knn_gather

import pdb



'''
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

xi, yi, zi = sample_spherical(64)
x = torch.Tensor(xi)
y = torch.Tensor(yi)
z = torch.Tensor(zi)
grid = torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)],0).unsqueeze(0).view(1,1,3,-1)/10
grid = torch.transpose(grid,3,2)
grid = torch.zeros_like(grid)
'''




class PolyPCN(BaseModel):
    def __init__(self):
        super().__init__()

        self.grid_size=8
        self.num_dense = 8192
        self.latent_dim = 512
        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        if True:
            self.Conv1_p = torch.nn.Parameter(torch.randn(3,6))
            self.Conv1_w = nn.Linear(3, 64)

            self.Conv2_p = torch.nn.Parameter(torch.randn(64,6))
            self.Conv2_w = nn.Linear(64, 128)

            self.Conv3_p = torch.nn.Parameter(torch.randn(128,6))
            self.Conv3_w = nn.Linear(128, 256)

            self.Conv4_p = torch.nn.Parameter(torch.randn(256,6))
            self.Conv4_w = nn.Linear(256, 512)
            




            self.in1 = nn.InstanceNorm1d(64,affine=True)
            self.in2 = nn.InstanceNorm1d(128,affine=True)
            self.in3 = nn.InstanceNorm1d(256,affine=True)
            self.in4 = nn.InstanceNorm1d(512,affine=True)
            
            #self.bn5 = nn.BatchNorm1d(1024)
            #self.bn2 = nn.BatchNorm1d(64)
            #self.bn0 = nn.BatchNorm1d(1024)
            
            #self.drop1 = nn.Dropout(p=0.2)
            #self.drop2 = nn.Dropout(p=0.5)
         
            self.mlp = nn.Sequential(
                nn.Linear(self.latent_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 3 * self.num_coarse)
            )

            self.final_conv = nn.Sequential(
                #nn.Conv1d(1024 + 3 + 2, 512, 1)
                nn.Conv1d(512 + 3 + 2, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 3, 1)
            )
            a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
            b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
            
            self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)            

            

        else:
            print("Check model name in config.json")
            raise

    def forward(self, input):
        def _closeness_constraint(netcoeff):  # (num_funcs, 6)
            # 6 parameters
            B = torch.zeros((netcoeff.shape[0], 3, 3), device='cuda')
            triu_idcs = torch.triu_indices(row=3, col=3, offset=0).to('cuda')
            B[:, triu_idcs[0], triu_idcs[1]] = netcoeff  # vector to upper triangular matrix
            B[:, triu_idcs[1], triu_idcs[0]] = netcoeff  # B: symm. matrix
            A = torch.bmm(B, B)  # A = B**2  // A: symm. positive definite (num_funcs, 3,3)

            # [1, x, y, x^2, xy, y^2] 
            p4coeff = torch.zeros((netcoeff.shape[0], 6), device='cuda')
            p4coeff[:, 0] = A[:, 0,0]  # 1
            p4coeff[:, 3] = A[:, 1,1]  # x^2
            p4coeff[:, 5] = A[:, 2,2]  # y^2

            p4coeff[:, 1] = A[:, 1,0]+A[:, 0,1]  # x
            p4coeff[:, 2] = A[:, 2,0]+A[:, 0,2]  # y
            p4coeff[:, 4] = A[:, 1,2]+A[:, 2,1]  # xy
            return p4coeff

        B, _, _ = input.shape

        if True:
            Conv1_p = _closeness_constraint(self.Conv1_p)
            Conv2_p = _closeness_constraint(self.Conv2_p)
            Conv3_p = _closeness_constraint(self.Conv3_p)
            Conv4_p = _closeness_constraint(self.Conv4_p)


            
            
            CONV = [Conv1_p,self.Conv1_w,Conv2_p,self.Conv2_w,Conv3_p,self.Conv3_w,Conv4_p,self.Conv4_w]
            IN = [self.in1,self.in2,self.in3,self.in4]


            feat = network(input, CONV, IN)
            #x = torch.mean(x,1)


            # decoder
            coarse = self.mlp(feat).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
            point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
            point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)
            seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
            seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

            feature_global = feat.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
            feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
            fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
            #return coarse.contiguous(), fine.contiguous()
            
            return fine.contiguous()            




if __name__ == "__main__":

    a = torch.rand(4,3,3096).to('cuda:0')
    net = PolyNet().to('cuda:0')
    output = net(a)
