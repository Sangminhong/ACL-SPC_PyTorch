import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import random
from utils.find_Nearest_Neighbor import find_NN as find_NN
from utils.output_xyz import output_xyz
import pdb
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
import time
from scipy.io import loadmat
import os.path as osp
import pickle
from collections import defaultdict
import logging

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

def transform_points(points,matrix):
    points = np.matmul(points,np.linalg.inv(matrix[:3,:3]))+matrix[:3, 3]
    return points


def resample(pc,N):
    ind = torch.randint(0,pc.shape[0],(N,))
    return pc[ind]

class NetDataset(data.Dataset):
    def __init__(self, args, root, dataset_name='shapenetv1_dpc/', 
            num_points=1548, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        
        self.root = os.path.join(root, dataset_name)
        self.split = split
        if args.class_name == 'plane':
            self.cls_name = "02691156"
        elif args.class_name == 'car':
            self.cls_name = "02958343"
        elif args.class_name == 'chair':
            self.cls_name = "03001627"

        if split=="test":
           metadata = open(self.root+'/splits/'+self.cls_name+'_test.txt','r').readlines()
           self.metadata = [x.split('\n')[0] for x in metadata]
           self.partial = h5py.File(self.root+ self.cls_name+ '_data.h5','r')
           self.gt = h5py.File(self.root+ self.cls_name+ '_gt.h5','r')
           
        elif split=="train":
           metadata = open(self.root+'/splits/'+self.cls_name+'_train.txt','r').readlines()
           self.metadata = [x.split('\n')[0] for x in metadata]  
           self.partial = h5py.File(self.root+ self.cls_name+ '_data.h5','r')  
           self.gt = h5py.File(self.root+ self.cls_name+ '_gt.h5','r')       
        self.cls_sizes = len(self.metadata)
        self.indices = self._indices_generator()    



    def __getitem__(self, item):
        if self.split =="train":
            frame = torch.randint(0,5,(1,))[0].numpy()
            trans = torch.from_numpy(np.asarray(self.partial[self.metadata[item]][str(frame)]['extrinsic'])).float()  
            partial_tmp = torch.from_numpy(np.asarray(self.partial[self.metadata[item]][str(frame)]['points'])).float() 
        
            partial_tmp = partial_tmp-trans[:3,3] 
            partial_tmp = torch.matmul(partial_tmp,trans[:3,:3])
            partial = resample(partial_tmp,3096)
        
            gt = torch.from_numpy(np.asarray(self.gt[self.metadata[item]]['gt'])).float() 
            gt = resample(gt,8192)
            R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)

            return partial, gt, R
            
        
        elif self.split =="test":
            partial_test, gt_test, R_test = [], [], []
            frames = [0,1,2,3,4]        
            for i in range(len(frames)):
                trans = torch.from_numpy(np.asarray(self.partial[self.metadata[item]][str(frames[i])]['extrinsic'])).float()  
                partial_tmp = torch.from_numpy(np.asarray(self.partial[self.metadata[item]][str(frames[i])]['points'])).float() 
        
                partial_tmp = partial_tmp-trans[:3,3] 
                partial_tmp = torch.matmul(partial_tmp,trans[:3,:3])
                partial = resample(partial_tmp,3096)
                gt = torch.from_numpy(np.asarray(self.gt[self.metadata[item]]['gt'])).float() 
                gt = resample(gt,8192)
                R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)  
                partial_test.append(partial.unsqueeze(0))         
                gt_test.append(gt.unsqueeze(0))
                R_test.append(R.unsqueeze(0))
            partial_test = torch.cat(partial_test,0)
            gt_test = torch.cat(gt_test,0)
            R_test = torch.cat(R_test,0)
            return partial_test, gt_test, R_test    


    def __len__(self):
        return self.cls_sizes

    def _indices_generator(self):
        indices = np.arange(self.cls_sizes)
        return indices.astype(int)


class SemanKITTIDataset:
    def __init__(self, split, directory, transform=None, nb_pts_thresh=None):
        split_map = {
        'train': (0, 1, 2, 3, 4, 5, 6, 7, 9, 10),
        'val': (8,),
        }
        self.directory = directory
        self.split = split
        self.transform = transform
        self.nb_pts_thresh = nb_pts_thresh
        self.split_map = split_map
        # data
        self._load_dataset(split)
        logger = logging.getLogger(__name__)
        logger.info(str(self))

    def _load_dataset(self, split):
        self.data = []
        for sequence_id in self.split_map[split]:
            fname = osp.join(self.directory, '{:02d}.pkl'.format(sequence_id))
            with open(fname, 'rb') as f:
                instances = pickle.load(f)
                print('Load', fname)
                for instance in instances:
                    token = '{:02d}_{}'.format(sequence_id, instance['instance_id'])
                    if token == '08_394':
                        print('Skip bad scans')
                        continue
                    for scan in instance['scans']:
                        if self.nb_pts_thresh and scan['points'].shape[0] < self.nb_pts_thresh:
                            continue
                        scan['model_id'] = token
                        # scan['token'] = token
                        scan['frame_id'] = scan['scan_id']
                        scan['gt'] = instance['gt2']  # use partial groundtruth
                        self.data.append(scan)

        # mapping model to frames
        self.token_to_index = defaultdict(list)
        self.model_to_frames = defaultdict(list)
        for idx, data in enumerate(self.data):
            self.token_to_index[data['model_id']].append(idx)
            self.model_to_frames[data['model_id']].append(data['frame_id'])
        self.model_ids = list(self.token_to_index.keys())

    def __getitem__(self, index):

        data = self.data[index]
        out_dict = dict()

        partial_points_vehicle = data['points'].astype(np.float32, copy=True)
        complete_points_object = data['gt'].astype(np.float32, copy=True)
        vehicle2object = data['vehicle2object'].copy()


        partial_points_lidar = transform_points(partial_points_vehicle,vehicle2object)
        partial_points_lidar = resample(partial_points_lidar,1024)
        R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)
        center = (np.max(complete_points_object,0)+np.min(complete_points_object,0))/2
        
        partial_points_lidar = partial_points_lidar-center
        complete_points_object = complete_points_object-center

        if self.split=="train":
           complete_points_object = resample(complete_points_object,8192)
           return torch.from_numpy(partial_points_lidar).float(), torch.from_numpy(complete_points_object).float(), R.float() 
        elif self.split=="val":      
           return torch.from_numpy(partial_points_lidar).float(), torch.from_numpy(complete_points_object).float(), R.float()


    def __len__(self):
        return len(self.data)



class PCNDataset(data.Dataset):
    def __init__(self, args, root, dataset_name='completion3d/', 
            num_points=1548, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        self.root = os.path.join(root, dataset_name)
        self.split = split
        
        if split=="test":
           metadata = open(self.root+'test.list','r').readlines()
           self.metadata = [x.split('\n')[0] for x in metadata]
           print(self.metadata[0])
           
        elif split=="val":
           metadata = open(self.root+'val.list','r').readlines()
           self.metadata = [x.split('\n')[0] for x in metadata]
           
        elif split=="train":
           metadata = open(self.root+'train.list','r').readlines()
           self.metadata = [x.split('\n')[0] for x in metadata]           
        self.cls_sizes = len(self.metadata)
        self.indices = self._indices_generator()


    def __getitem__(self, item):
        
        partial = torch.from_numpy(np.asarray(h5py.File(self.root+ self.split + '/partial/' + self.metadata[item] + '.h5','r')['data'])).float()    
        R = torch.cat([random_pose().unsqueeze(0) for i in range(8)],0)
            
        #label,name
        if self.split=="test":            
            return partial, self.metadata[item].split('/')[1]
        else:
            gt = torch.from_numpy(np.asarray(h5py.File(self.root+ self.split + '/gt/' + self.metadata[item] + '.h5','r')['data'])).float()
            return partial, gt, R, self.metadata[item].split('/')[1]

    def __len__(self):
        return self.cls_sizes

    def _indices_generator(self):
        indices = np.arange(self.cls_sizes)
        return indices.astype(int)




