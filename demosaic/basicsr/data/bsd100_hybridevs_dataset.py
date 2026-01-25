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

#Thanks to DemosaicFormer for code.
import shutil
from PIL import Image
import os
import numpy as np
import glob
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


class BSD100HybridEVSDataset(Dataset):

    def getHW(self):
        img = imageio.imread(self.files[0])
        H = img.shape[0]
        W = img.shape[1]
        return H,W

    def __init__(self,opt):

        self.opt = opt
        globpath = "datasets/bsd100/bicubic_4x/*/HR/*.png"
        self.files = sorted(glob.glob(globpath))      
        print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,i):

        img = imageio.imread(self.files[i])[:,:,0:3]
        img = torch.tensor(img)
        img = (img/255).float()

        img = img.permute(2,0,1)

        img = img.cpu()

        lq = torch.tensor(evsQuadBayerSampler(img.permute(1,2,0).numpy())).permute(2,0,1)
        
        return {"lq":lq,"gt":img}