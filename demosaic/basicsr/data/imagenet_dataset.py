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
import torchvision
import random
import torchvision.transforms.functional as TF
import tqdm
import torch.nn.functional as F

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

class ImageNetDataset(Dataset):

    def getHW(self):
        img = imageio.imread(self.files[0])
        H = img.shape[0]
        W = img.shape[1]
        return H,W

    def __init__(self,opt):

        self.opt = opt
        path = "/home/featurize/data"
        globpath = path+"/ILSVRC2012_img_train/*/*.JPEG"
        read_files = glob.glob(globpath)
        self.cfa_format = "bayer"
        if "bayer" in opt:
            cfas = {1:"bayer",2:"quad_bayer",3:"nona_bayer"}
            self.cfa_format = cfas[opt["bayer"]]
        self.crop = torchvision.transforms.RandomCrop(128)
        self.cfa = torch.tensor(get_cfa(128,128,self.cfa_format)).permute(2,0,1)
        self.files = []
        for file in tqdm.tqdm(read_files):
            img = imageio.imread(file)
            if (len(img.shape)==3 and img.shape[0]>=128 and img.shape[1]>=128 and img.shape[2]==3):
                img = torch.tensor(img)
                img = img.permute(2,0,1)
                img = self.crop(img)
                img = img*self.cfa
                self.files.append(file)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):

        cfa = self.cfa

        img = imageio.imread(self.files[i])
        img = torch.tensor(img)
        img = img.permute(2,0,1)
        img = self.crop(img)
            
        img = (img/255).float()            
        k = int(torch.rand(1)*4)
        img = torch.rot90(img,k=k,dims=(1,2))
        hflip = torch.rand(1)<0.5
        vflip = torch.rand(1)<0.5
        if (hflip):
            img = TF.hflip(img)
        if  (vflip):
            img = TF.vflip(img)

        img = img.cpu()
        
        cfa_img = img*cfa
        return {"lq":cfa_img.float(),"gt":img}