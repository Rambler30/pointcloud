from pathlib import Path
import random
import pyrootutils
root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))
from src.datasets.GDUT_config import *
from src.datasets.base import BaseDataset
import numpy as np
from src.data import Data
import torch
import os.path as osp
from src.utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb, shift_point_cloud,\
    jitter_point_cloud, rotate_perturbation_point_cloud_with_normal, \
    shuffle_points, shuffle_data

from hydra import compose, initialize
import hydra
from hydra.utils import instantiate

__all__ = ['GDUT']

def read_grss_points(path, xyz=True, rgb=True, intensity=False, semantic=True, instance=False) -> Data:
    root, path = osp.split(path)
    data_name = 'split_'+ path.split('.')[0].split('_')[1] + '.txt'
    data_path = osp.join(root, data_name)
    cloud = np.loadtxt(data_path)
    cloud, _ = shuffle_data(cloud)
    labeled_points_indexs = ~(cloud[:,-1]==0)
    cloud = cloud[labeled_points_indexs]
    cloud[cloud[:,-1]==20, -1] = 0
    if path.split('.')[0].split('_')[0] == 'perturbate':
        xyz_data = perturbate_data(cloud)
    elif path.split('.')[0].split('_')[0] == 'shift':
        xyz_data = shift_data(cloud)
    else:
        xyz_data = normal_data(cloud)

    rgb_data = np.array(cloud[:, 3:6],dtype='uint8')
    y_data = np.array(cloud[:,-1], dtype='int')
    o_data = np.array(cloud[:,-1], dtype='int')
    xyz_data = torch.from_numpy(xyz_data) if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(rgb_data)) if rgb else None
    y_data = torch.from_numpy(y_data) if semantic else None
    o_data = torch.from_numpy(o_data) if instance else None
    data = Data(pos=xyz_data, rgb=rgb_data, y=y_data, o=o_data)
    return data

def normal_data(cloud):
    xyz_data = np.array(cloud[:, :3],dtype='float32')
    return np.array(xyz_data,dtype='float32')

def perturbate_data(cloud):
    perturbate_xyz = rotate_perturbation_point_cloud_with_normal(cloud[:,:3])
    return np.array(perturbate_xyz,dtype='float32')

def shift_data(cloud):
    shift_xyz = shift_point_cloud(cloud[:,:3])
    return np.array(shift_xyz,dtype='float32')

class GDUT(BaseDataset):
    def __init__(self,  *args, fold = [], **kwargs):
        self.fold = fold
        super().__init__(*args, val_mixed_in_train=True, **kwargs)

    @property
    def class_names(self) -> list:
        return CLASS_NAME
    
    @property
    def num_classes(self):
        return len(INV_OBJECT_LABEL)
    
    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        return read_grss_points(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=False)
    
    @property
    def all_base_cloud_ids(self):   
        return {
            'train': [f'{name}_{i}' for i in range(10) if i not in  self.fold for name in ['split', 'perturbate', 'shift']],
            'val': [f'{name}_{i}' for i in range(10) if i not in  self.fold for name in ['split', 'perturbate', 'shift']],
            'test': [f'{name}_{i}' for i in self.fold for name in ['split', 'perturbate', 'shift']]
        }