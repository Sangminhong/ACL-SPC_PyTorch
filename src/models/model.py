
import torch
import torch.nn as nn

class mlp(nn.Module):

    def __init__(self):
        super(mlp, self).__init__()
        n_layers=6 
        n_in= 2048
        n_out= 4096
        layers = []
        for i in range(n_layers-1):
            layers.append(nn.Linear(n_in, n_in))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(n_in, n_out))
        self.layers =  nn.Sequential(*layers)    


    def forward(self, pc):
        out = self.layers(pc)
        x = torch.linspace(-1, 1, steps=16, device=out.device)
        y = torch.linspace(-1, 1, steps=16, device=out.device)
        z = torch.linspace(-1, 1, steps=16, device=out.device)
        x, y, z = torch.meshgrid(x,y,z)
        grid = torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)],0).unsqueeze(0).view(1,3,-1)
       
        return out
