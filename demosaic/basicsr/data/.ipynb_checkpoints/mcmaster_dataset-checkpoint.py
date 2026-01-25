import os
import glob
import numpy as np
from torch.utils.data import Dataset
import imageio
import cv2
import torch
import time
import h5py
import h5pickle

def generate_qxq_bayer(H,W,Q):
    filter = np.zeros((H,W,3))

    for i in range(H):
        for j in range(W):
            filter[i][j][0] = 1 if ((i//Q)%2==0 and (j//Q)%2==0) else 0
            filter[i][j][2] = 1 if ((i//Q)%2==1 and (j//Q)%2==1) else 0
            filter[i][j][1] = 1 if (filter[i][j][0]==0 and filter[i][j][2]==0) else 0
    
    return filter

def get_cfa(H,W,name):
    if ("bayer" in name):
        name_to_dim = {"bayer":1,"quad_bayer":2,"nona_bayer":3}
        return generate_qxq_bayer(H,W,name_to_dim[name])
    else:
        return NotImplementedError

class McMasterDataset(Dataset):

    def getHW(self):
        img = imageio.imread(self.files[0])
        H = img.shape[0]
        W = img.shape[1]
        return H,W

    def __init__(self,opt):

        self.opt = opt
        globpath = "datasets/mcmaster/*.tif"
        self.files = glob.glob(globpath)
        self.cfa_format = "bayer"
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):

        img = imageio.imread(self.files[i])[:,:,0:3]
        img = torch.tensor(img)
        img = (img/255).float()

        img = img.permute(2,0,1)

        img = img.cpu()

        cfa = torch.tensor(get_cfa(img.shape[1],img.shape[2],self.cfa_format)).permute(2,0,1)
        
        cfa_img = img*cfa
        
        return {"lq":cfa_img.float(),"gt":img}