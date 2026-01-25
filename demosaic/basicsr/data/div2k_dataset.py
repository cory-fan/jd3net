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

def evsQuadBayerSampler(image, use_evsavg=False):
    img = image.copy()

    # evs pix
    if use_evsavg:
        img[1::4, 1::4, 0] = (
            img[::4, ::4, 0] + img[::4, 1::4, 0] + img[1::4, ::4, 0]
        ) / 3
        img[3::4, 3::4, 2] = (
            img[2::4, 2::4, 2] + img[2::4, 3::4, 2] + img[3::4, 2::4, 2]
        ) / 3
        # print(img[1::4,1::4, 0], img[3::4,3::4, 2])
    else:
        img[1::4, 1::4, 0] = 0
        img[3::4, 3::4, 2] = 0

    # Quad R
    img[::4, ::4, 1:3] = 0
    img[1::4, 1::4, 1:3] = 0
    img[::4, 1::4, 1:3] = 0
    img[1::4, ::4, 1:3] = 0

    # Quad B
    img[3::4, 2::4, 0:2] = 0
    img[3::4, 3::4, 0:2] = 0
    img[2::4, 3::4, 0:2] = 0
    img[2::4, 2::4, 0:2] = 0

    # Quad G12
    img[1::4, 2::4, ::2] = 0
    img[1::4, 3::4, ::2] = 0
    img[::4, 2::4, ::2] = 0
    img[::4, 3::4, ::2] = 0

    # Quad G21
    img[2::4, 1::4, ::2] = 0
    img[3::4, 1::4, ::2] = 0
    img[2::4, ::4, ::2] = 0
    img[3::4, ::4, ::2] = 0

    return img

def generate_qxq_bayer(H,W,Q):
    filter = np.zeros((H,W,3))

    for i in range(H):
        for j in range(W):
            filter[i][j][0] = 1 if ((i//Q)%2==0 and (j//Q)%2==0) else 0
            filter[i][j][2] = 1 if ((i//Q)%2==1 and (j//Q)%2==1) else 0
            filter[i][j][1] = 1 if (filter[i][j][0]==0 and filter[i][j][2]==0) else 0
    
    return filter

def generate_quad_hybridevs_bayer(H,W):
    filter = np.ones((H,W,3))
    return evsQuadBayerSampler(filter)

def get_cfa(H,W,name):
    if ("bayer" in name):
        name_to_dim = {"bayer":1,"quad_bayer":2,"nona_bayer":3}
        return generate_qxq_bayer(H,W,name_to_dim[name])
    elif (name == "hybridevs"):
        return generate_quad_hybridevs_bayer(H,W)

class CustomValDataset(Dataset):

    def getHW(self):
        img = imageio.imread(self.files[0])
        H = img.shape[0]
        W = img.shape[1]
        return H,W

    def __init__(self,opt):

        SIZE = 10

        self.opt = opt
        path = "/home/featurize/data"
        globpath = path+"/DIV2K_valid_HR/DIV2K_valid_HR/*.png"
        self.files = glob.glob(globpath)[:SIZE]
        self.cfa_format = "bayer"
        if "bayer" in opt:
            cfas = {1:"bayer",2:"quad_bayer",3:"nona_bayer",4:"hybridevs"}
            self.cfa_format = cfas[opt["bayer"]]

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