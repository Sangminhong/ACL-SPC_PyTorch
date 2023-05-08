from symbol import arglist
import torch 
import torchvision
import numpy as np
import argparse
import os
from train import trainer
import sys
from models.polynet import PolyNet
import torch.utils.data as data
import open3d as o3d
from utils.output_xyz import output_xyz
import torch.optim as optim
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from loss.loss import ChamferLoss, chamfer_distance_naive, chamfer_distance_kdtree
from loss.loss import MSELoss 
from utils.Logger import Logger
from tqdm import tqdm
from torch_scatter import scatter_min, scatter_mean

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='Point cloud Completion')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--experiment_id', type=str, default= 'demo', help='experiment id ')
    parser.add_argument('--input', type=str, default='0.ply', help = 'filename of inpute file')
    parser.add_argument('--model_filename', type=str, default='/mnt/disk2/mchiash2/experiments/bunny/models/ShapeNet_5000.pkl', help='filename of pretrained model')
    parser.add_argument('--dataset_directory', type=str, default='/mnt/disk2/mchiash2/bunny/inputs/train/', help='The directory that contains dataset')
    parser.add_argument('--output_filename', type=str, default='./demo_output.ply', help='output filename')
    parser.add_argument('--class_name', type=str, default='plane', help='class of dataset', choices = ['plane', 'car', 'chair', 'table'])
    parser.add_argument('--fine_tune', type = str2bool, default = False)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')                   
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=1, metavar='N',
                        help='Save snapshot interval ')                
    parser.add_argument('--num_points', type=int, default=8192, metavar='N',
                        help='Num_points before removing (Original num_points)')
    args = parser.parse_args()
    return args

def random_pose():
    angle_y = -(np.random.uniform() * np.pi/3 - np.pi/9)
    angle_z = np.random.uniform() * 2 * np.pi
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.matmul(Rz,Ry)
    R = torch.from_numpy(R).float()
    return R

def resample(pc):
    ind = torch.randint(0,pc.shape[1],(3096,))
    return pc[:,ind]

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

class FineTuneDataset(data.Dataset):
    def __init__(self, args, 
            num_points=1548, load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        self.train_root = os.path.join(args.dataset_directory)
        self.train_data = []

        for filename in os.listdir(self.train_root):
            self.train_data.append(os.path.join(self.train_root, filename))  


    def __getitem__(self, item):
        
        partial_pc = o3d.io.read_point_cloud(self.train_data[item])
        partial = np.asarray(partial_pc.points)   
        partial = torch.from_numpy(partial)
        R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0) 
        #R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)     
        #partial, gt = torch.from_numpy(partial), torch.from_numpy(gt) #, partial_tmp.shape[0]
        return partial.type(torch.FloatTensor), R
        

    def __len__(self):
        return len(self.train_data)

class FineTuner(object):
    def __init__(self, args, path_to_model, dataset):
        self.train_dataset = dataset
        self.batch_size = args.batch_size
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size= self.batch_size,
                shuffle=True,
                num_workers= 1
            )
        self.num_points = args.num_points                           
        self.device = torch.device("cuda")
        self.model = PolyNet().cuda()
        self.path_to_model = path_to_model
        #self.find_NN_batch = find_NN_batch().to(self.device)
        self.parameter1 = self.model.parameters()
        self.optimizer1 = optim.Adam(self.parameter1, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer1, step_size=200, gamma=0.5)
        self.gamma = 0.5
        self.loss1 = chamfer_distance_kdtree 
        self.loss2 = MSELoss()    
        self.epochs = args.epochs
        self.snapshot_interval = 10
        self.experiment_id = args.experiment_id
        self.snapshot_root = '/mnt/disk2/mchiash2/experiments/%s' % self.experiment_id
        self.save_dir = os.path.join(self.snapshot_root, 'models/')
        self.num_syn = 8
        sys.stdout = Logger(os.path.join(self.snapshot_root, 'log.txt'))
        self.args =args
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }        

    
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
        if epoch == 0:
            self.model.load_state_dict(torch.load(path_to_model))
            self.model.train()

        for iter, data in enumerate(tqdm(self.train_loader)):
            
            inputs, R = data
            inputs = inputs.to(self.device)
            R = R.to(self.device)
            inputs = torch.transpose(inputs, 1,2)   
            self.optimizer1.zero_grad()
            output2= self.model(inputs)

            # losses of the input
            dist21, dist22 = self.loss1(inputs, output2)
            loss1 = 0.9*torch.mean(dist21)+0.1*torch.mean(dist22)        
            loss2 = 0
            out_final = [output2]
            syns, cnt_syn = synthesize(output2.unsqueeze(1),R)

            for i in range(self.num_syn):
                o_syn2 = self.model(syns[:,i].detach())
                out_final.append(o_syn2)
                loss2 += self.loss2(o_syn2,output2)/self.num_syn

            out_final = sum(out_final)/(self.num_syn+1)


            


            loss = (loss1+10*loss2)
            loss.backward()
            self.optimizer1.step()
            loss_buf.append(loss.detach().cpu().numpy())
            del loss
        
        self.scheduler.step()

        
        if epoch%1 ==0:
            self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}')

        return out_final





if __name__ == '__main__':
    args = get_parser()
    # Define the path to the pretrained model
    path_to_model = args.model_filename
   
    # Define the input to the model
    incomplete_pc = torch.Tensor((o3d.io.read_point_cloud(args.input)).points).cuda()
    incomplete_pc = incomplete_pc.transpose(0,1).unsqueeze(0)

    # Load the pretrained model
    model = PolyNet().cuda() 
    model.load_state_dict(torch.load(path_to_model))


    # Set the model to evaluation mode
    #model.eval()
    if args.fine_tune:
        dataset = FineTuneDataset(args)
        finetune = FineTuner(args, path_to_model, dataset)

        for epoch in tqdm(range(args.epochs)):
            out_final = finetune.train_epoch(epoch=epoch)

        output_xyz(out_final.transpose(1,2)[0], args.output_filename)
    else:
        # Pass the input through the model
        complete_pc = model(incomplete_pc)
        output_xyz(complete_pc.transpose(1,2)[0], args.output_filename)


