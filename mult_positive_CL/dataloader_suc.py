import os
import torch
import glob
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, metadata,metadata2,metadata3,metadata4,root_dir, transform = None):
        # self.metadata = pd.read_csv(metadata).iloc[:480]
        self.metadata = pd.read_csv(metadata)

#        self.label = pd.read_csv(label)
        self.metadata2 = np.loadtxt(metadata2).reshape(1,-1)
        self.metadata3 = np.loadtxt(metadata3).reshape(1,-1)
        self.metadata4 = np.loadtxt(metadata4).reshape(1, -1)
#        self.metadata3 = np.loadtxt(metadata3).reshape(1,-1)
        self.root_dir = root_dir
#        print('self.root_dir',self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        #sd_idx=int(self.metadata3[0,idx])
        path = self.metadata.iloc[idx, 0]

        im=Image.open(self.root_dir+path+'.png')
        cp_idx=self.metadata2[0,idx]
        human_idx = self.metadata3[0,idx]
        geo_idx = self.metadata4[0,idx]
        try:
            path_cp = self.metadata.iloc[int(cp_idx), 0]
            path_human = self.metadata.iloc[int(human_idx), 0]
            path_geo = self.metadata.iloc[int(geo_idx), 0]
        except IndexError as e:
            print(f"The value of x is: {cp_idx}")


        im_cp=Image.open(self.root_dir+path_cp+'.png')
        im_human = Image.open(self.root_dir + path_human + '.png')
        im_geo = Image.open(self.root_dir + path_geo + '.png')
        if self.transform:
            sample = self.transform(im)
            sample_cp=self.transform(im_cp)
            sample_human = self.transform(im_human)
            sample_geo = self.transform(im_geo)
        return sample, sample_cp, sample_human, sample_geo

# class MyDataset_3(Dataset):
#     def __init__(self, metadata,metadata2,metadata3,root_dir, transform = None):
#         self.metadata = pd.read_csv(metadata)
# #        self.label = pd.read_csv(label)
#         self.metadata2 = np.loadtxt(metadata2).reshape(1,-1)
#         self.metadata3 = np.loadtxt(metadata3).reshape(1,-1)
#         self.root_dir = root_dir
# #        print('self.root_dir',self.root_dir)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.metadata)
#        # return self.metadata3.shape[1]
#
#     def __getitem__(self, idx):
#         #sd_idx=int(self.metadata3[0,idx])
#         path = self.metadata.iloc[idx, 0]
# #        print('idx',idx)
# #        print('path',path)
#         im=Image.open(self.root_dir+path+'.png')
# #        label=self.label.iloc[idx,0]
#         cp_idx=self.metadata2[0,idx]
# #        print('cp_idx',cp_idx)
#         path_cp = self.metadata.iloc[int(cp_idx), 0]
#         im_cp=Image.open(self.root_dir+path_cp+'.png')
#
#         p_idx=self.metadata3[0,idx]
# #        print('cp_idx',cp_idx)
#         path_p = self.metadata.iloc[int(p_idx), 0]
#         im_p=Image.open(self.root_dir+path_p+'.png')
#
#         if self.transform:
#             sample = self.transform(im)
#             sample_cp=self.transform(im_cp)
#             sample_p=self.transform(im_p)
#
#         return sample,sample_cp,sample_p
#
