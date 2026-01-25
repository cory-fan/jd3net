"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
SaiKiran Tedla
Abhijith Punnappurath
Luxi Zhao
Michael S. Brown

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
import random
import pickle
import os
import random
import torch.nn.functional as F

def build_pattern(order = "RGGB", filter_size = 1, rows = 300, cols = 300, pattern_dict = {"R":0, "G": 1, "B": 2}):
    pattern_tile = torch.empty((filter_size*2, filter_size*2), dtype=torch.int64)
    for row in range(2):
        for col in range(2):
            pattern_index = row*2 + col
            channel_letter = order[pattern_index]
            pattern_channel = pattern_dict[channel_letter]
            pattern_tile[row*filter_size:(row+1)*filter_size, col*filter_size:(col+1)*filter_size] = pattern_channel
    pattern = torch.tile(pattern_tile, (rows//(filter_size*2), cols//(filter_size*2)))
    return pattern

def construct_mosaic(image, order = "RGGB", filter_size = 1):
    pattern = build_pattern(order=order, filter_size=filter_size, rows=image.shape[0], cols=image.shape[1])
    out = torch.gather(input=image, dim=2, index=pattern.unsqueeze_(dim=-1)).squeeze()
    return pattern, out


class HDD_Dataset(Dataset):
    def __init__(self, opt):
                 # imgs_dir, split, pattern, remosaic, iso=3200, patch_size=120, patch_num_per_image=1, max_trdata=0, hard_patches_percentile = 0, noisy_viz=False, dropout=0, mask_interpolation=False):
        self.opt = opt
        self.imgs_dir = opt['imgs_dir']
        self.patch_size = opt['patch_size']
        self.patch_num_per_image = opt['patch_num_per_image']
        self.split = opt['split']
        self.pattern = opt['pattern']
        self.iso = opt['iso']
        self.noisy_viz=opt['noisy_viz']
        self.dropout = opt['dropout']
        self.remosaic = False
        self.mask_interpolation = opt['mask_interpolation']
        assert self.split in ['train', 'test', 'val', 'mining', 'test_full_size', 'all'], "Unknown split"
        assert self.pattern in ['single', 'quad', 'nona', 'single_quad', 'single_quad_nona', 'quad_single', 'nona_single', 'random'], "Unknown pattern"

        if self.pattern == "single":
            self.filter_size = [1]
        elif self.pattern == "quad":
            self.filter_size = [2]
        elif self.pattern == "nona":
            self.filter_size = [3]
        elif self.pattern == "single_quad":
            self.filter_size = [1,2]
        elif self.pattern == "single_quad_nona":
            self.filter_size = [1,2,3]
        elif self.pattern == "quad_single":
            self.filter_size = [2,1]
        elif self.pattern == "nona_single":
            self.filter_size = [3,1]

        #leave 18 out due to the feathers
        #I don't know why this is done but the original codebase has it
        split_info ={"train": [3,4,5,6,9,10,11,12,13,14], "val": [1,2], "test": [7, 8, 15, 16, 17], "test_full_size": [7, 8, 15, 16, 17], 'mining':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
        

        logging.info('Loading images information...')

        imgs_dir = self.imgs_dir
        split = self.split

        self.imgfiles = []
        if self.iso in [100, 6400, 12800, 100000]:
            for scene in split_info[self.split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_gt.npy'))
        elif self.iso == 400:
            for scene in split_info[self.split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso400.npy'))
        elif self.iso == 800:
            for scene in split_info[self.split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso800.npy'))
        elif self.iso == 1600:
            for scene in split_info[self.split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso1600.npy'))
        elif self.iso==3200:
            for scene in split_info[self.split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso3200.npy'))
   

        if opt['hard_patches_percentile'] != 0:
            # open a file, where you stored the pickled data
            hpp = opt['hard_patches_percentile']
            hardpatches_file = os.path.join(imgs_dir, f'hardpatches{hpp:.2f}.pkl')
            with open(hardpatches_file, 'rb') as f:
                # dump information to that file
                hard_patches = set(pickle.load(f))

                #filter so I only keep hard_patches
                hard_patch_list = []
                for img in self.imgfiles:
                    scene_view = img.split('/')[-3:-1]
                    scene_view = '/'.join(scene_view)
                    img_row_col = img.split('/')[-1].split('_')[0:5]
                    img_row_col = '_'.join(img_row_col)
                    if f'{scene_view}/{img_row_col}' in hard_patches:
                            hard_patch_list.append(img)
                
                self.imgfiles = hard_patch_list
        
        self.imgfiles = sorted(self.imgfiles) #sort the files by name

        random.seed(0)

        if self.split == "train" or self.split == "val": #this shuffle will be the same because I set the seed somewhere else
            random.shuffle(self.imgfiles)

        logging.info(f'Creating {split} dataset with {len(self.imgfiles)} examples')


    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, img, patch_size, patch_coords, remosaic, pattern):

        if patch_size != -1: #val/test is -1
            img = img[patch_coords[0]:patch_coords[0]+patch_size, patch_coords[1]:patch_coords[1]+patch_size, :]
 
        # HWC to CHW
        img_trans = img.permute((2, 0, 1))
        img_trans = img_trans / (13496)

        return img_trans

    def __getitem__(self, i):
        

        gt_ext = 'gt.npy'
        img_file = self.imgfiles[i]
        in_img_npy = np.load(img_file).astype(np.uint16)

        in_img = torch.from_numpy(in_img_npy.astype(np.float32))



        if self.noisy_viz:
            in_img = in_img
        else:
            in_img_all_mosaics = torch.empty((in_img.shape[0], in_img.shape[1], len(self.filter_size)*4), dtype=in_img.dtype)
            for i, fs in enumerate(self.filter_size):
                pattern, mosaic = construct_mosaic(in_img, order="RGGB", filter_size=fs)
                pattern = pattern.long()


                if self.split == "train":
                    dropout = torch.rand((1))*self.dropout #randomly choose a value between 0 and self.dropout
                    rand = torch.rand(mosaic.shape)
                    mask = rand < dropout
                else: #if test
                    dropout = self.dropout
                    torch.manual_seed(hash(img_file))
                    rand = torch.rand(mosaic.shape[0]*mosaic.shape[1])
                    num_dead_pixels = int(dropout * mosaic.shape[0] * mosaic.shape[1])
                    values, indices = torch.topk(rand, num_dead_pixels) #always get this number of dead pixels
                    
                    rows = torch.div(indices, mosaic.shape[0], rounding_mode='trunc')
                    cols = torch.div(indices, mosaic.shape[1], rounding_mode='trunc')

                    mask = torch.zeros(mosaic.shape, dtype=torch.bool)
                    mask[rows, cols] = True

                    if self.mask_interpolation:
                        for dp_num in range(num_dead_pixels):
                            #compute a 7x7 filter (this will be the same for everything)
                            R = torch.arange(-3, 4)
                            C = torch.arange(-3, 4)

                            mC, mR = torch.meshgrid(C,R, indexing='ij')
                            gauss_sigma = 3
                            filter = torch.exp(-(((mR.type(torch.float32)) ** 2 + (mC.type(torch.float32)) ** 2) / (2 * gauss_sigma ** 2))) * (1 / (2 * np.pi * gauss_sigma ** 2))
                            #get pixels around my central dead pixel
                            mosaic_pad = F.pad(input=mosaic, pad=(3, 3, 3, 3), mode='constant', value=0)
                            crop = mosaic_pad[rows[dp_num]: rows[dp_num]+7, cols[dp_num]: cols[dp_num]+7]

                            #mask pixels that are outside row/col boundaries to fitler weight of 0
                            R = torch.arange(rows[dp_num]-3, rows[dp_num]+4)
                            C = torch.arange(cols[dp_num]-3,cols[dp_num] + 4)
                            mR, mC = torch.meshgrid(R,C, indexing='ij')

                            filter[mR<0] = 0
                            filter[mR>mosaic.shape[0]-1] = 0
                            filter[mC<0] = 0
                            filter[mC>mosaic.shape[1]-1] = 0
                            #set filter weights for things that are not the same color to 0
                            
                            color = pattern[rows[dp_num], cols[dp_num], 0]

                            mR = torch.clamp(mR, 0, mosaic.shape[0] - 1)
                            mC = torch.clamp(mC, 0, mosaic.shape[1] - 1)
                            
                            filter[pattern[mR,mC,0] != color] = 0

                            #set other dead pixels filter to 0 (don't put other dead pixels in the filter), because I am updating the mosaic directly
                            #if this works this will also set the filter weight of the central pixel
                            for odp_num in range(num_dead_pixels):
                                if (rows[odp_num]-rows[dp_num] >= -3 and rows[odp_num]-rows[dp_num] <= 3):
                                    if (cols[odp_num]-cols[dp_num] >= -3 and cols[odp_num]-cols[dp_num] <=3):
                                        filter[rows[odp_num]-rows[dp_num]+3, cols[odp_num]-cols[dp_num]+3] = 0

                            #normalize filter

                            filter = filter/torch.sum(filter)

                            #compute interpolated color

                            out_color = torch.sum(crop*filter)
                            mosaic[rows[dp_num], cols[dp_num]] = out_color

                if not self.mask_interpolation:
                    mosaic[mask] = 0 #kill these pixels
                    
                pattern = pattern[:,:,0]
                if self.mask_interpolation:
                    R_pattern = (pattern == 0).long()*13496
                    G_pattern = (pattern == 1).long()*13496
                    B_pattern = (pattern == 2).long()*13496
                else:
                    R_pattern = torch.logical_and((pattern == 0), ~mask).long()*13496
                    G_pattern = torch.logical_and((pattern == 1), ~mask).long()*13496
                    B_pattern = torch.logical_and((pattern == 2), ~mask).long()*13496
                in_img_all_mosaics[:,:, i*4] = mosaic
                in_img_all_mosaics[:,:, i*4+1] = R_pattern
                in_img_all_mosaics[:,:, i*4+2] = G_pattern
                in_img_all_mosaics[:,:, i*4+3] = B_pattern
            in_img = in_img_all_mosaics

        # get image size
        w, h, _= in_img.shape

        # get ground truth images
        parts = img_file.split('_')
        base_name = ''
        for k in range(len(parts) - 1):
            base_name = base_name + parts[k] + '_'
        gt_file = base_name + gt_ext
        gt_img_npy = np.load(gt_file).astype(np.uint16)
        gt_img = torch.from_numpy(gt_img_npy.astype(np.float32))

        # if self.iso == 100000:
        #     gt_img = gt_img/low_light_scale # low light


        if self.split == "train" or self.split == "all":
            patch_x = 0
            patch_y = 0

            in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), self.remosaic, self.pattern)
            gt_img_patches = self.preprocess(gt_img, self.patch_size, (patch_x, patch_y), False, self.pattern)

            return {'lq': in_img_patches, 'gt': gt_img_patches.repeat(in_img_patches.shape[0]//4,1,1), 'img_file': img_file}
        elif self.split == "test_full_size":
            in_img = self.preprocess(in_img, -1, (-1, -1), self.remosaic, self.pattern)
            gt_img = self.preprocess(gt_img, -1, (-1, -1), False, self.pattern) 
            return {'lq': in_img, 'gt': gt_img.repeat(in_img.shape[0]//4,1,1), 'img_file': img_file}
        else:
            in_img = self.preprocess(in_img, -1, (-1, -1), self.remosaic, self.pattern)
            gt_img = self.preprocess(gt_img, -1, (-1, -1), False, self.pattern)
            return {'lq': in_img, 'gt': gt_img.repeat(in_img.shape[0]//4,1,1), 'img_file': img_file}