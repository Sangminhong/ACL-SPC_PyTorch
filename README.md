# ACL-SPC_PyTorch

This repository contains the official code to reproduce the results from the paper:

**ACL-SPC: Adaptive Closed-Loop system for Self-Supervised Point Cloud Completion** (CVPR 2023)

\[[arXiv](https://arxiv.org/abs/2303.01979)\] \[[presentation]()\] 

![architecture](https://github.com/Sangminhong/ACL-SPC_PyTorch/blob/master/assets/NewFramework-1.png)

## Installation
Clone this repository into any place you want.
```
git clone https://github.com/Sangminhong/ACL-SPC_PyTorch.git
cd ACL-SPC_PyTorch
```
### Dependencies
* Python 3.8.5
* PyTorch 1.7.1
* numpy
* h5py
* numba
* scikit-learn
* open3d
* torchsummary
* pytorch3d
* KNN-CUDA
* pykdtree
* torch_scatter

## Quick Start
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Expriments

#### Pretrained model
Download `model_best.pth` from this [link](https://drive.google.com/drive/folders/1tG3hBXtroHe4iXHb5W8XIfQ8YJEeS3Tp?usp=sharing) and save them.
#### NOTE: The pretrained model is updated at April. 24th 2023. 

You can now go to src folder and test our ACL-SPC:
```
CUDA_VISIBLE_DEVICES=0 python main.py --experiment_id {experiment id} --dataset_name {dataset} --class_name {plane/car/chair/table}  
```


### Citation
If you find our code or paper useful, please consider citing:
```
@inproceedings{Hong2023ACLSPC,
  title={ACL-SPC: Adaptive Closed-Loop system for Self-Supervised Point Cloud Completion},
  author={Sangmin Hong and Mohsen Yavartanoo and Reyhaneh Neshatavar and Kyoung Mu Lee},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
