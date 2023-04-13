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




class PolyNet(BaseModel):
    def __init__(self):
        super().__init__()

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
            layers = []
            layers.append(nn.Linear(512, 1024))
            layers.append(nn.ReLU(inplace=True))
            for i in range(1):
                 layers.append(nn.Linear(1024, 1024))
                 layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(1024, 8192*3))          
            self.decoder = nn.Sequential(*layers)

            

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


        if True:
            Conv1_p = _closeness_constraint(self.Conv1_p)
            Conv2_p = _closeness_constraint(self.Conv2_p)
            Conv3_p = _closeness_constraint(self.Conv3_p)
            Conv4_p = _closeness_constraint(self.Conv4_p)


            
            
            CONV = [Conv1_p,self.Conv1_w,Conv2_p,self.Conv2_w,Conv3_p,self.Conv3_w,Conv4_p,self.Conv4_w]
            IN = [self.in1,self.in2,self.in3,self.in4]


            feat = network(input, CONV, IN)
            #x = torch.mean(x,1)
            

            x = self.decoder(feat).view(-1,3,8192)

        return x


if __name__ == "__main__":

    a = torch.rand(4,3,3096).to('cuda:0')
    net = PolyNet().to('cuda:0')
    output = net(a)
