import os
import sys
from tqdm import tqdm

import math
import numpy as np
import pickle
from numpy.random import RandomState
import trimesh
import matplotlib.pyplot as plt


class RealWorldPointsDataset:
    def __init__(self, mesh_dir, batch_size=50, npoint=2048, shuffle=True, split='train', random_seed=None):
        '''
        part_point_cloud_dir: the directory contains the oringal ply point clouds
        batch_size:
        npoint: a fix number of points that will sample from the point clouds
        shuffle: whether to shuffle the order of point clouds
        normalize: whether to normalize the point clouds
        split: 
        extra_ply_point_clouds_list: a list contains some extra point cloud file names, 
                                     note that only use it in test time, 
                                     these point clouds will be inserted in front of the point cloud list,
                                     which means extra clouds get to be tested first
        random_seed: 
        '''
    
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mesh_dir = mesh_dir
        self.npoint = npoint
        self.split = split
        self.random_seed = random_seed

        # make a random generator
        self.rand_gen = RandomState(self.random_seed)
        #self.rand_gen = np.random

        # list of meshes
        self.meshes = self._read_all_meshes(self.mesh_dir) # a list of trimeshes
        self._preprocess_meshes_as_ShapeNetV2(self.meshes) # NOTE: using different processing...

        self.point_clouds = self._pre_sample_points(self.meshes)

        self.reset()

    def _shuffle_list(self, l):
        self.rand_gen.shuffle(l)
    
    def _preprocess_meshes_old(self, meshes):
        '''
        currently, just normalize all meshes, according to the height
        also, snap chairs to the ground
        '''

        max_height = -1

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])

            if height > max_height:
                max_height = height
            
        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            scale_factor = height / max_height

            bbox_center = np.mean(bbox.vertices, axis=0)
            bbox_center[1] = height / 2.0 # assume that the object is alreay snapped to ground

            trans_v = -bbox_center 
            trans_v[1] += mesh.bounding_box.extents[1]/2.
            mesh.apply_translation(trans_v) # translate the bottom center bbox center to ori

            mesh.apply_scale(scale_factor) # do scaling

        return 0
    
    def _preprocess_meshes(self, meshes):
        '''
        assume the input mesh has already been snapped to the ground
        1. normalize to fit within a unit cube
        2. center the bottom center to the original
        '''

        for mesh in meshes:
            bbox = mesh.bounding_box # 8 pts
            height = np.max(bbox.vertices[:,1])
            extents = mesh.bounding_box.extents.copy()
            extents[1] = height

            scale_factor = 1.0 / np.amax(extents)

            bbox_center = np.mean(bbox.vertices, axis=0)

            trans_v = -bbox_center 
            trans_v[1] = 0 # assume already snap to the ground, so do not translate along y
            mesh.apply_translation(trans_v) # translate the center bbox bottom to ori

            mesh.apply_scale(scale_factor)
        
    def _preprocess_meshes_as_ShapeNetV2(self, meshes):
        '''
        the input meshes are pre-aligned facing -z and snapped onto the ground.
        then, make the diagonal length of the axis aligned bounding box around the shape is equal to 1
        center object bbox center to the original
        '''
        for mesh in meshes:
            
            pts_min = np.amin(mesh.vertices, axis=0)
            pts_min[1] = 0 # using the real height
            pts_max = np.amax(mesh.vertices, axis=0)
            diag_len = np.linalg.norm(pts_max - pts_min)

            scale_factor = 1.0 / diag_len

            bbox_center = (pts_max + pts_min) / 2.0

            trans_v = -bbox_center 
            mesh.apply_translation(trans_v) # translate the center of bbox to ori

            mesh.apply_scale(scale_factor)
        return
    
    def _read_all_meshes(self, mesh_dir):
        meshes_cache_filename = os.path.join(os.path.dirname(mesh_dir), 'meshes_cache_%s.pickle'%(self.split))
        
        if os.path.exists(meshes_cache_filename):
            #print('Loading cached pickle file: %s'%(meshes_cache_filename))
            p_f = open(meshes_cache_filename, 'rb')
            mesh_list = pickle.load(p_f)
            p_f.close()
        else:
            split_filename = os.path.join(os.path.dirname(mesh_dir), os.path.basename(mesh_dir)+'_%s_split.pickle'%(self.split))
            with open(split_filename, 'rb') as pf:
                mesh_name_list = pickle.load(pf)
            mesh_filenames = []
            for mesh_n in mesh_name_list:
                mesh_filenames.append(os.path.join(mesh_dir, mesh_n))
            mesh_filenames.sort() # NOTE: sort the file names here!

            print('Reading and caching...')
            mesh_list = []
            for mn in tqdm(mesh_filenames):
                m_fn = os.path.join(mesh_dir, mn)
                mesh = trimesh.load(m_fn)
            
                mesh_list.append(mesh)
            
            p_f = open(meshes_cache_filename, 'wb')
            pickle.dump(mesh_list, p_f)
            print('Cache to %s'%(meshes_cache_filename))
            p_f.close()

        return mesh_list

    def _pre_sample_points(self, meshes):
        presamples_cache_filename = os.path.join(os.path.dirname(self.mesh_dir), 'presamples_cache_%s.pickle'%(self.split))
        if os.path.exists(presamples_cache_filename):
            #print('Loading cached pickle file: %s'%(presamples_cache_filename))
            p_f = open(presamples_cache_filename, 'rb')
            points_list = pickle.load(p_f)
            p_f.close()
        else:
            print('Pre-sampling...')
            points_list = []
            for m in tqdm(meshes):
                samples, _ = trimesh.sample.sample_surface_even(m, self.npoint * 10)
                points_list.append(np.array(samples))

            p_f = open(presamples_cache_filename, 'wb')
            pickle.dump(points_list, p_f)
            p_f.close()

            print('Pre-sampling done and cached.')

        return points_list

    def reset(self):
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_list(self.meshes)

    def has_next_batch(self):
        num_batch = np.floor(len(self.meshes) / self.batch_size) + 1
        if self.batch_idx < num_batch:
            return True
        return False
    
    def next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        end_idx = (self.batch_idx+1) * self.batch_size

        data_batch = np.zeros((self.batch_size, self.npoint, 3))
        for i in range(start_idx, end_idx):

            if i >= len(self.point_clouds):
                i_tmp = i % len(self.point_clouds)
                pc_cur = self.point_clouds[i_tmp]
            else:
                pc_cur = self.point_clouds[i] # M x 3
                
            choice_cur = self.rand_gen.choice(pc_cur.shape[0], self.npoint, replace=True)
            idx_cur = i % self.batch_size
            data_batch[idx_cur] = pc_cur[choice_cur, :]

        self.batch_idx += 1
        return data_batch

    def get_npoint(self):
        return self.npoint

