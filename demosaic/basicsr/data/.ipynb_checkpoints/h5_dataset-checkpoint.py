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

class CustomTrainDataset(Dataset):

    def __init__(self,opt):
        super().__init__()
        if ("dataroot" not in opt):
            self.f = h5pickle.File("/home/featurize/data/h5/complete.h5")
            self.len = 276000
        else:
            self.f = h5pickle.File(opt["dataroot"])
            self.len = len(self.f['complete'])

    def __len__(self):
        return self.len

    def __getitem__(self,i):
        retobj = {"lq":self.f['complete'][i][0],"gt":self.f['complete'][i][1],"i":i};
        if (torch.sum(torch.isnan(torch.tensor(retobj['lq'])))>0):
            print("ERROR ERROR ERROR %i"%i)
        return retobj