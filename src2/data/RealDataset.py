from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from numpy.random import RandomState
import h5py
import random
from tqdm import tqdm
import pickle
import glob
from utils.io import read_ply_xyz, read_ply_from_file_list
from utils.pc_transform import swap_axis
from data.real_dataset import RealWorldPointsDataset
from plyfile import PlyData

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

class RealDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC
    """
    def __init__(self, args, class_choice, split):
        #self.dataset = args.dataset
        #self.dataset_path = args.dataset_path
        self.root = args.root
        self.dataset = args.dataset_name
        self.random_seed = 0
        self.rand_gen = RandomState(self.random_seed)

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            if self.dataset == 'ScanNet':
                REALDATASET = RealWorldPointsDataset(self.root+'real_data/combined/data/scannet_v2_'+class_choice+'s_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=split, random_seed=0)
            elif self.dataset == 'MatterPort':
                if split in ['train', 'trainval']:
                    REALDATASET = RealWorldPointsDataset(self.root+'real_data/combined/data/MatterPort_v1_'+class_choice+'_Yup_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split='test', random_seed=0)
                else:
                    REALDATASET = RealWorldPointsDataset(self.root+'real_data/combined/data/MatterPort_v1_'+class_choice+'_Yup_aligned/point_cloud', batch_size=6, npoint=2048,  shuffle=False, split=split, random_seed=0)
            elif self.dataset == 'KITTI':
                if split in ['train']:
                    REALDATASET = KITTIDataset(self.root+'real_data/combined/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_train/')
                elif split in ['test', 'val']:
                    REALDATASET = KITTIDataset(self.root+'real_data/combined/data/KITTI_frustum_data_for_pcl2pcl/point_cloud_val/')
            input_ls = REALDATASET.point_clouds 
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [np.float32(swap_axis(itm, swap_mode='n210')) for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
    
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            choice = self.rand_gen.choice(self.input_ls[index].shape[0], 2048, replace=True)
            input_pcd = self.input_ls[index][choice,:]
            R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)
            return (input_pcd, stem, R)    
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  


class KITTIDataset():
    def __init__(self, load_path):
        self.point_clouds = []
        file_list = glob.glob(load_path + '*.ply')
        total_num = len(file_list)
        for i in range(total_num):
            file_name = load_path + str(i) + '.ply'
            ply_file = PlyData.read(file_name)
            pc = np.array([ply_file['vertex']['x'], ply_file['vertex']['y'], ply_file['vertex']['z']])
            pc = np.transpose(pc,(1,0))
            self.point_clouds.append(pc)
        return



