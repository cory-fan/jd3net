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

def quadBayerSampler(image):
    img = image.copy()

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


def read_bin(img_path):
    """
    Read bin data
    """
    img_data = np.fromfile(img_path, dtype=np.uint16)
    w = int(img_data[0])
    h = int(img_data[1])
    assert w * h == img_data.size - 2
    quad = np.clip(img_data[2:].reshape([h, w]).astype(np.float32), 0, 1023)
    return quad


def crop_and_save(image, image_name, output_dir, crop_size=512, stride=256):

    i = 0
    height = image.shape[0]
    width = image.shape[1]
    print(width, height)
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            if y + crop_size > height:
                y0 = height - 512
                y1 = height
            else:
                y0 = y
                y1 = y + crop_size
            if x + crop_size > width:
                x0 = width - 512
                x1 = width
            else:
                x0 = x
                x1 = x + crop_size

            crop = image[y0:y1, x0:x1, :]

            crop = Image.fromarray(crop)
            cropped_image_name = f"{image_name}_{i}.png"

            crop.save(os.path.join(output_dir, cropped_image_name))
            i = i + 1


def Sampler(img):

    img = np.asarray(img)
    img = img.copy()
    img = np.stack((img,) * 3, axis=-1)
    # print(img.shape)
    H, W, _ = img.shape
    H4, W4 = H // 4, W // 4
    img = quadBayerSampler(img)

    return img

class MIPIDataset(Dataset):

    def __init__(self,opt):

        data_path = "/home/featurize/data"

        lq_globpath = data_path+"/input/*.bin"
        gt_globpath = data_path+"/gt/*.png"

        self.lq_files = sorted(glob.glob(lq_globpath))
        self.gt_files = sorted(glob.glob(gt_globpath))

        self.patch_size = opt['gt_size']
        self.concat_filter = opt['concat_filter']
        
        self.d = opt['crop']+1

        assert(len(self.lq_files)==len(self.gt_files))

    def __len__(self):
        return len(self.lq_files)

    def generate_hybridevs_filter(self,H,W):
        img = torch.concat((torch.ones(H,W,3),torch.zeros(H,W,1)),dim=-1)
        
        img[1::4, 1::4, 0] = 0
        img[3::4, 3::4, 2] = 0
        img[1::4, 1::4, 3] = 1
        img[3::4, 3::4, 3] = 1
        
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
        
        return img.permute(2,0,1)

    def randomly_crop_and_pad(self,x):
        height_crop = int(torch.rand(1)*self.d)
        width_crop = int(torch.rand(1)*self.d)
        C,H,W = x.shape
        cropped_x = x[:,:H-height_crop,:W-width_crop]
        filt = self.generate_hybridevs_filter(cropped_x.shape[1],cropped_x.shape[2])
        padded_x = F.pad(cropped_x, (0,width_crop,0,height_crop))
        filt = F.pad(filt,(0,width_crop,0,height_crop))
        return padded_x, filt

    def __getitem__(self,i):
        lq_file = self.lq_files[i]
        gt_file = self.gt_files[i]


        inp_quad = torch.tensor(Sampler(read_bin(lq_file))).permute(2,0,1) / 1023
        gt = torch.tensor(imageio.imread(gt_file)).permute(2,0,1) / 255

        # if (random.random()<0.25):

        #     concat_im = torch.concat((inp_quad,gt),dim=0)
    
        #     _,H,W = gt.shape
            
        #     img = TF.crop(concat_im,int(random.random()*(H-128)//4)*4,int(random.random()*(W-128)//4)*4,128,128)
    
        #     lq = img[:3,:,:]
        #     gt = img[3:,:,:]

        # else:

        _,H,W = gt.shape

        crop_height = int(random.random()*(H-self.patch_size)//4)*4
        crop_width = int(random.random()*(W-self.patch_size)//4)*4

        img = TF.crop(gt,crop_height,crop_width,self.patch_size,self.patch_size)          
        k = int(torch.rand(1)*4)
        img = torch.rot90(img,k=k,dims=(1,2))
        hflip = torch.rand(1)<0.5
        vflip = torch.rand(1)<0.5
        if (hflip):
            img = TF.hflip(img)
        if  (vflip):
            img = TF.vflip(img)

        img,filt = self.randomly_crop_and_pad(img)

        img = img.cpu()

        gt = img
        if (self.concat_filter):
            lq = torch.concat((img*filt[:3,:,:],filt),dim=0)
        else:
            lq = img*filt[:3,:,:]
        

        return {"lq":lq,"gt":gt}
        
        
        