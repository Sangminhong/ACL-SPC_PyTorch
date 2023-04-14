# ACL-SPC_PyTorch

This repository contains the official code to reproduce the results from the paper:

**ACL-SPC: Adaptive Closed-Loop system for Self-Supervised Point Cloud Completion**

\[[arXiv](https://arxiv.org/abs/2303.01979)\] \[[presentation]()\] 


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

Or you can just run the below commands to set the environment.
```
conda env create --file environment.yml
conda activate ACL_SPC
```

## Quick Start
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Expriments

#### Pretrained model
Download `config.json` and `model_best.pth` from this [link]() and save them in `` folder.
#### NOTE: The pretrained model is updated at March. 1st 2023. 

You can now go to src folder and test our ACL-SPC by:
```
python test.py 
```

or you can train it by yourself as follows:
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
