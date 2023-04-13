from symbol import arglist
import torch 
import numpy as np
import argparse
import os
from train import trainer
from  train_semkitti import trainer as trainer_semkitt
from train_real import trainer as trainer_real
from pretrain import pretrainer
import torch.multiprocessing as mp
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_parser():
    parser = argparse.ArgumentParser(description='Point cloud Completion')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--experiment_id', type=str, default= 'experiment1', help='experiment id ')
    parser.add_argument('--dataset_name', type=str, default='ShapeNet', help='The name of the dataset', choices = ['ShapeNet', 'SemanticKITTI', 'MatterPort', 'ScanNet', 'KITTI'])
    parser.add_argument('--root', type=str, default='/mnt/disk2/mchiash2/', help='The directory that contains dataset and experiment')
    parser.add_argument('--class_name', type=str, default='plane', help='class of dataset', choices = ['plane', 'car', 'chair', 'table'])
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
    parser.add_argument('--pretrain', type = bool, default = False, choices = [True, False])                    
    parser.add_argument('--num_points', type=int, default=8192, metavar='N',
                        help='Num_points before removing (Original num_points)')
    parser.add_argument('--resume', type=str, default=None, metavar='N',
                        help='checkpoint address')
    parser.add_argument('--remov_ratio', type=int, default=8, metavar='N',
                        help='How much part of point cloud is goint to be removed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    ExpNum_dir = os.path.join("/mnt/disk2/mchiash2/experiments/", args.experiment_id )
    if os.path.isdir(ExpNum_dir) is False:
       os.mkdir(ExpNum_dir)
    if os.path.isdir(os.path.join(ExpNum_dir, "models")) is False:
       os.mkdir(os.path.join(ExpNum_dir, "models"))
    isdir = os.path.isdir("../outputs/"+str(args.experiment_id))
    if isdir is False:      
        os.mkdir("../outputs/"+str(args.experiment_id))
        os.mkdir("../outputs/"+str(args.experiment_id)+'/train')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/gt')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc/0')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc/1')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc/2')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc/3')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/ipc/4')
        
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc/0')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc/1')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc/2')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc/3')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/cpc/4')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse/0')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse/1')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse/2')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse/3')
        os.mkdir("../outputs/"+str(args.experiment_id)+'/test/coarse/4')
    np.save('../outputs/'+str(args.experiment_id)+'/result.npy',100*np.ones(4))

    if args.pretrain is True:
        pretrain = pretrainer(args)
        pretrain.run(args)        

    elif (args.dataset_name == 'ShapeNet'): 
        train = trainer(args)
        train.run(args)

    elif args.dataset_name == 'SemanticKITTI':
        train = trainer_semkitt(args)
        train.run(args)

    elif (args.dataset_name == 'MatterPort') or (args.dataset_name == 'ScanNet') or (args.dataset_name == 'KITTI'):
        train = trainer_real(args)
        train.run(args)


