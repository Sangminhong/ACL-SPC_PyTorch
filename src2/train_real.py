import torch 
import torch.nn as nn
import numpy as np
import sys
import os
from data.RealDataset import RealDataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.polynet import PolyNet
from loss.loss import  chamfer_distance_kdtree, distChamfer, directed_hausdorff   
from loss.loss import MSELoss 
import torch.optim as optim
from tqdm import tqdm
import time
from utils.find_Nearest_Neighbor import find_NN
from utils.find_Nearest_Neighbor import find_NN_batch
from utils.Logger import Logger
import util
import random
import pdb
from utils.output_xyz import output_xyz
from pytorch3d.ops.knn import knn_points
from torch_scatter import scatter_min, scatter_mean
import open3d as o3d


def resample(pc, N):
    ind = torch.randint(0,pc.shape[1],(N,))
    return pc[:,ind]
    
def downsample(point,num):
   outs = []
   for i in range(point.shape[0]):
       ind = torch.randint(0,point.shape[2],(num,))
       out = point[i,:,ind].unsqueeze(0)
       outs.append(out)
   return torch.cat(outs,0)

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def synthesize(pcs_input,R=None):
    pcs = torch.transpose(pcs_input,2,3)
    pcs = torch.matmul(pcs,R)
    res = 16
    inds = torch.clamp(torch.round(res*(pcs+0.5)),0,res).long()
    inds = (res*inds[:,:,:,1]+inds[:,:,:,2]).long()        
    new_pcs = []
    cnt = []
    for i in range(pcs.shape[0]):
        new_pcs_i = []
        cnt_i = []
        for j in range(pcs.shape[1]):
            pc = pcs[i,j]
            ind = inds[i,j]
            un, inv = torch.unique(ind,return_inverse=True)
            out, argmin = scatter_min(pc[:,0], inv, dim=0)
            new_pc = pcs_input[i,0][:,argmin]
            new_pc = resample(new_pc, 2048).unsqueeze(0)
            new_pcs_i.append(new_pc)
            cnt_i.append(argmin.shape[0])
        new_pcs_i = torch.cat(new_pcs_i,0).unsqueeze(0)
        cnt.append(cnt_i)
        new_pcs.append(new_pcs_i)
    new_pcs = torch.cat(new_pcs,0)
    return(new_pcs), cnt

    
def sample_spherical(npoints, ndim=3):
    x = np.linspace(-0.5,0.5,npoints)
    y = np.linspace(-0.5,0.5,npoints)
    z = np.linspace(-0.5,0.5,npoints)
    X,Y,Z = np.meshgrid(x,y,z)
    vec = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    return vec
    
def grids(N=128):
    grid_sphere = torch.Tensor(sample_spherical(N)).unsqueeze(0).view(1,3,-1)
    return grid_sphere


class trainer(object):

    def __init__(self, args):


        self.train_dataset = RealDataset(args, args.class_name, 'train')
        self.test_dataset = RealDataset(args, args.class_name, 'test')
        self.batch_size =args.batch_size
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size= self.batch_size,
                shuffle=False,
                num_workers= 1
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size= 4,
                shuffle=False,
                num_workers= 1
            )
        self.num_points = args.num_points                           
        self.device = torch.device("cuda")
        self.model = PolyNet()
        self.find_NN_batch = find_NN_batch().to(self.device)
        self.parameter1 = self.model.parameters()
        self.optimizer1 = optim.Adam(self.parameter1, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer1, step_size=200, gamma=0.5)
        self.gamma = 0.5
        self.loss1 = chamfer_distance_kdtree 
        self.loss2 = MSELoss()  
        self.haus = directed_hausdorff
        self.eval_CD = distChamfer
        self.epochs = args.epochs
        self.snapshot_interval = 10
        self.experiment_id = args.experiment_id
        self.snapshot_root = args.root + 'experiments/%s' % self.experiment_id
        self.save_dir = os.path.join(self.snapshot_root, 'models/')
        self.dataset_name = args.dataset_name
        self.num_syn = 8
        self.grid_sphere = grids(N=8).to(self.device)
        sys.stdout = Logger(os.path.join(self.snapshot_root, 'log.txt'))
        self.args =args

    

    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
            
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        #torch.save(new_state_dict_implicit, save_dir + "_" + str(epoch) + '_implicit.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")



    def train_epoch(self, epoch):

        loss_buf=[]
        UCD_buf = []
        UHD_buf = []
        self.model=self.model.to(self.device)        
        if self.args.resume!=None and epoch==int(self.args.resume.split(os.sep)[-1][9:-4]):
           self.model.load_state_dict(torch.load(self.args.resume))
           self.model.train()

        
        for iter, data in enumerate(tqdm(self.test_loader)):
            
            inputs, index, R = data
            inputs = inputs.to(self.device)
            R = R.to(self.device)            
            inputs = torch.transpose(inputs, 1,2)   
            self.optimizer1.zero_grad()

            output2 = self.model(inputs)
            dist21, dist22 = self.loss1(inputs, output2)
            loss1 = 0.8*torch.mean(dist21)+0.2*torch.mean(dist22) #+ self.loss2(output2,output) 
            loss2 = 0
            out_final = [output2]
            syns, cnt_syn = synthesize(output2.unsqueeze(1),R)

   
            for i in range(self.num_syn):
                o_syn2 = self.model(syns[:,i].detach())
                out_final.append(o_syn2)
                loss2 += self.loss2(o_syn2,output2)/self.num_syn
                
            out_final = sum(out_final)/(self.num_syn+1)


            output_xyz(torch.transpose(output2, 1,2)[0], '../outputs/'+ str(self.experiment_id)+ '/train/cpc_'+str(iter)+'.ply')
            output_xyz(torch.transpose(inputs, 1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/train/ipc_'+str(iter)+'.ply')
            output_xyz(torch.transpose(syns[:,0], 1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/train/v_'+str(iter)+'.ply')


            dist1, dist2, _,_ = self.eval_CD(inputs.transpose(1,2), output2.transpose(1,2))
            UCD = dist1.mean()
            UHD = self.haus(inputs,output2)
            loss = (loss1+10*loss2)
            loss.backward()
            self.optimizer1.step()
            loss_buf.append(loss.detach().cpu().numpy())
            UCD_buf.append(UCD.detach().cpu().numpy())
            UHD_buf.append(UHD.detach().cpu().numpy())
            del loss
        
        self.scheduler.step()


        if epoch>100 or epoch%10 ==9:
            self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}')
        print(f'Epoch {epoch+1}: UCD {np.mean(UCD_buf)}')
        print(f'Epoch {epoch+1}: UHD {np.mean(UHD_buf)}')
        
        return np.mean(loss_buf), np.mean(UCD_buf), np.mean(UHD_buf)


    def test_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            
            chamf = []
            Dist1 = []
            Dist2 = []
            OUT_o2 = []
            for iter, test_data in enumerate(tqdm(self.train_loader)):
                test_pc, test_gt, R = test_data

                test_pc = test_pc.to(self.device)
                test_gt = test_gt.to(self.device)
                test_pc = torch.transpose(test_pc, 1,2)
                test_gt = torch.transpose(test_gt, 1,2)
                output_xyz(torch.transpose(test_gt,1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/test/gt/'+str(iter)+'.ply')
                output_xyz(torch.transpose(test_pc,1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/test/ipc/'+str(iter)+'.ply')

        return chamf
    


    def run(self,args):

        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }

        best_loss = 1000000000
        best_UCD = 1000000000
        best_UHD = 1000000000
        print("Training start~")
        print(f'exp_num: {self.args.experiment_id} epoch: {self.args.epochs} initial_Lr: {self.args.lr} scheduler_gamma: {self.gamma} exp_description: {self.args.exp_name}' )
        start_time = time.time()
        
        epoch_init = 0
        if args.resume!=None:
            epoch_init = int(args.resume.split(os.sep)[-1][9:-4])
        for epoch in tqdm(range(epoch_init, self.epochs)):  
             
            loss, UCD, UHD = self.train_epoch(epoch)
            #if epoch%10==0:

            if UCD < best_UCD:
                best_UCD = UCD

            if UHD < best_UHD:
                best_UHD = UHD
            
            
            print(f'Epoch {epoch+1}: best_UCD {best_UCD}')
            print(f'Epoch {epoch+1}: best_UHD {best_UHD}')
                


        end_time = time.time()
        self.train_hist['total_time'].append(end_time- start_time)

        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")




if __name__ == '__main__':
    print("Testing this file")



    

