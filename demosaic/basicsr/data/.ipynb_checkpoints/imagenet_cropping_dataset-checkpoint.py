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

class ImageNetCropDataset(Dataset):

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
        self.crop = torchvision.transforms.RandomCrop(129)
        self.cfa = torch.tensor(get_cfa(129,129,self.cfa_format)).permute(2,0,1)
        self.files = []
        for file in tqdm.tqdm(read_files):
            img = imageio.imread(file)
            if (len(img.shape)==3 and img.shape[0]>=129 and img.shape[1]>=129 and img.shape[2]==3):
                img = torch.tensor(img)
                img = img.permute(2,0,1)
                img = self.crop(img)
                img = img*self.cfa
                self.files.append(file)

    def randomly_crop_and_pad(self,x):
        height_crop = int(torch.rand(1)*3)
        width_crop = int(torch.rand(1)*3)
        C,H,W = x.shape
        cropped_x = x[:,:H-height_crop,:W-width_crop]
        cfa_filter = self.cfa[:,:H-height_crop,:W-width_crop]
        lq = torch.concat((cropped_x*cfa_filter,cfa_filter),dim=0)
        padded_x = F.pad(lq, (0,width_crop,0,height_crop))
        return padded_x, F.pad(cropped_x, (0, width_crop, 0, height_crop))
        

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

        lq,gt = self.randomly_crop_and_pad(img)

        return {"lq":lq.float(),"gt":gt}