import torch 
import torch.nn as nn
import numpy as np
import sys
import os
from data.data import SemanKITTIDataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.polynet import PolyNet
from loss.loss import ChamferLoss, chamfer_distance_naive, chamfer_distance_kdtree, directed_hausdorff
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
import time
from utils.output_xyz import output_xyz
from pytorch3d.ops.knn import knn_points
from torch_scatter import scatter_min, scatter_mean
import open3d as o3d


def resample(pc):
    ind = torch.randint(0,pc.shape[1],(1024,))
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
    res = 64
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
            new_pc = resample(new_pc).unsqueeze(0)
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
    grid_sphere = torch.Tensor(sample_spherical(N)).unsqueeze(0).view(1,3,-1)#torch.cat([xi.unsqueeze(0),yi.unsqueeze(0),zi.unsqueeze(0)],0).view(1,3,-1)
    return grid_sphere

class trainer(object):

    def __init__(self, args):
        self.train_dataset = SemanKITTIDataset(split='train', directory =args.root+'semantic_kitti_comp_v2/')
        self.test_dataset = SemanKITTIDataset(split='val', directory = args.root+'semantic_kitti_comp_v2/')
        self.batch_size = args.batch_size
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size= self.batch_size,
                shuffle=True,
                num_workers= 1
            )
        self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size= 1,
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
        self.epochs = args.epochs
        self.snapshot_interval = 10
        self.experiment_id = args.experiment_id
        self.snapshot_root = args.root+'experiments/%s' % self.experiment_id
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
                name = key[7:]  
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")


   

    def train_epoch(self, epoch):

        loss_buf=[]
        chamf_buf = []
        dist1_buf = []
        dist2_buf = []
        self.model=self.model.to(self.device)
        
        if self.args.resume!=None and epoch==int(self.args.resume.split(os.sep)[-1][9:-4]):
           self.model.load_state_dict(torch.load(self.args.resume))
           self.model.train()
    
        for iter, data in enumerate(tqdm(self.test_loader)):
            
            inputs, gt, R = data
      
            x_mean = 0.4362
            y_mean = 0.1870
            z_mean = 0.1421
            r_x = x_mean/(torch.max(gt[:,:,0],1)[0]).item()
            r_y = y_mean/(torch.max(gt[:,:,1],1)[0]).item()
            r_z = z_mean/(torch.max(gt[:,:,2],1)[0]).item()

            inputs[:,:,0] = -r_x*inputs[:,:,0]
            inputs[:,:,1] = r_y*inputs[:,:,1]
            inputs[:,:,2] = r_z*inputs[:,:,2]
            gt[:,:,0] = -gt[:,:,0]
  
            inputs = inputs.to(self.device)
            gt = gt.to(self.device)
            R = R.to(self.device)  

            inputs = torch.transpose(inputs, 1,2)
            gt = torch.transpose(gt, 1,2)     
            self.optimizer1.zero_grad()
            output2 = self.model(inputs)  
            dist21, dist22 = self.loss1(inputs, output2)


            loss1 = 0.9*torch.mean(dist21)+0.1*torch.mean(dist22) #+ self.loss2(output2,output)
            
            loss2 = 0
            out_final = [output2]
            syns, cnt_syn = synthesize(output2.unsqueeze(1),R)


            
            
            for i in range(self.num_syn):
                o_syn2 = self.model(syns[:,i].detach())
                out_final.append(o_syn2)
                loss2 += self.loss2(o_syn2,output2)/self.num_syn
                
                
            out_final = sum(out_final)/(self.num_syn+1)
            

            if iter%10==0:              
                output_xyz(torch.transpose(output2, 1,2)[0], '../outputs/'+ str(self.experiment_id)+ '/train/cpc_'+str(iter)+'.ply')
                #output_xyz(torch.transpose(indputs, 1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/train/ipc_'+str(iter)+'.ply')
                #output_xyz(torch.transpose(syns[:,0], 1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/train/v_'+str(iter)+'.ply')
                #output_xyz(torch.transpose(gt, 1,2)[0] , '../outputs/'+ str(self.experiment_id)+  '/train/gt_'+str(iter)+'.ply')
            
            
            R_xyz = torch.reciprocal(torch.Tensor([r_x, r_y, r_z])).to(self.device)
            output2 = R_xyz.mul(output2.transpose(1,2)).transpose(1,2)
            dist1, dist2 = self.loss1(inputs, output2) 
            chamf = torch.mean(dist1)+torch.mean(dist2)
            UCD = dist2.mean()
            UHD = self.haus(inputs, output2)
            loss = (loss1+10*loss2)
            loss.backward()
            self.optimizer1.step()
            dist1_buf.append(torch.mean(dist1).detach().cpu().numpy())
            dist2_buf.append(torch.mean(dist2).detach().cpu().numpy())
            chamf_buf.append(chamf.detach().cpu().numpy())

        self.scheduler.step()
        self._snapshot(epoch)        
        if epoch>100 or epoch%10 ==9:
        
            self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}')
        print(f'Epoch {epoch+1}: chamf {np.mean(chamf_buf)}')
        print(f'Epoch {epoch+1}: dist1 {np.mean(dist1_buf)}')
        print(f'Epoch {epoch+1}: dist2 {np.mean(dist2_buf)}')
        
        return np.mean(loss_buf)


    def test_epoch(self, epoch):
        self.model=self.model.to(self.device)
        if self.args.resume!=None and epoch==int(self.args.resume.split(os.sep)[-1][9:-4]):
           self.model.load_state_dict(torch.load(self.args.resume))        
        self.model.eval()

        with torch.no_grad():
            
            chamf = []
            Dist1 = []
            Dist2 = []
            OUT_o2 = []
            for iter, test_data in enumerate(tqdm(self.test_loader)):
                test_pc, test_gt, R = test_data
                test_pc[:,:,0] = -test_pc[:,:,0]
                test_gt[:,:,0] = -test_gt[:,:,0]
                test_pc = test_pc.to(self.device)
                test_gt = test_gt.to(self.device)
                test_pc = torch.transpose(test_pc, 1,2)
                test_gt = torch.transpose(test_gt, 1,2)
                test_output2 = self.model(test_pc)             
                dist1, dist2 = self.loss1(test_gt, test_output2)    
                chamf.append((torch.mean(dist1)+torch.mean(dist2)).cpu().numpy())


        return chamf
    



    def run(self,args):

        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print("Training start~")
        print(f'exp_num: {self.args.experiment_id} epoch: {self.args.epochs} initial_Lr: {self.args.lr} scheduler_gamma: {self.gamma} exp_description: {self.args.exp_name}' )
        start_time = time.time()
        
        epoch_init = 0
        if args.resume!=None:
            epoch_init = int(args.resume.split(os.sep)[-1][9:-4])
        for epoch in tqdm(range(epoch_init, self.epochs)):  
             
            loss = self.train_epoch(epoch)
            #if epoch%10==0:

            # save snapshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
            """    
            if test_loss < best_loss:
                best_loss = test_loss
                self._snapshot('best')
            """
    

        end_time = time.time()
        self.train_hist['total_time'].append(end_time- start_time)

        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")





