# %%
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

from skimage.io import imread, imsave
from cellpose import models, plot, utils
from cellpose import io

# %%
dirpath = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/train_rgb_c210_merge")
filelist = list(dirpath.joinpath('img').glob('*.png'))
datapath = dirpath.joinpath('data')
segpath = dirpath.joinpath('seg')
datapath.mkdir(parents=True, exist_ok=True)
segpath.mkdir(parents=True, exist_ok=True)

n = 3 
filelist_split = [filelist[i: i+n] for i in range(0, len(filelist), n)] 
for filepaths in filelist_split[:1]:
    print(filepaths)
    imgs = [imread(filepath) for filepath in filepaths]
    model = models.Cellpose(gpu = True, model_type = 'cyto2')
    masks, flows, styles, diams = model.eval(imgs, 
                      diameter = 150, 
                      channels = [1, 3], 
                      flow_threshold = 0.8,
                      cellprob_threshold = 0.0
                      )
    opdatapath = [datapath.joinpath(f'{filepath.stem}.npy') for filepath in filepaths]
    opsegpath = [segpath.joinpath(f'{filepath.stem}.png') for filepath in filepaths]
    io.masks_flows_to_seg(imgs, masks, flows, diams, opdatapath)
    io.save_to_png(imgs, masks, flows, opsegpath)
    

# %%    
seglist = list(dirpath.joinpath('seg').glob('*.png'))
for seglist in seglist[:1]:
    seg = imread(seglist)
    plt.imshow(seg)
    plt.show()
    seg_1 = seg == 1
    plt.imshow(seg_1)
    plt.show()


# %%

datalist = list(dirpath.joinpath('data').glob('*.npy'))

for datapath in datalist[:1]:
    print(datapath)
    data = np.load(datapath, allow_pickle=True).item()

    print(len(masks))
    mask_RGB = plot.mask_overlay(data['img'], data['masks'],
                                 # colors=np.array(data['colors'])
                                 )
    outlines = utils.outlines_list(masks)
    plt.imshow(img)
    for o in outlines:
        plt.plot(o[:, 0], o[:, 1], 'r') 
    plt.show()
    

# %%

def cellpose_seg(args):
    filepaths = [Path(x) for x in args.i] 
    opdir = Path(args.o)
    opdir.mkdir(parents=True, exist_ok=True)
    
    imgs = [imread(filepath) for filepath in filepaths]
    model = models.Cellpose(gpu = True, model_type = 'cyto2', device = args.gpu_device)
    masks, flows, styles, diams = model.eval(imgs, 
                      diameter = 150, 
                      channels = [1, 3], 
                      flow_threshold = 0.8,
                      cellprob_threshold = 0.0
                      )
    opdatapath = [opdir.joinpath(f'{filepath.stem}.npy') for filepath in filepaths]
    opsegpath = [opdir.joinpath(f'{filepath.stem}.png') for filepath in filepaths]
    io.masks_flows_to_seg(imgs, masks, flows, diams, opdatapath)
    io.save_to_png(imgs, masks, flows, opsegpath)
    
    return

def main():
    description = "merge three channls into one single RGB file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', type=str, required=True, nargs='+', help='input dir')
    parser.add_argument('-o', type=str, required=True, help='output dir')
    parser.add_argument('-g', '--gpu_device', type=int, default = '0', required=True, help='gpu device')    

    args = parser.parse_args()
    cellpose_seg(args) 
    return    

if __name__ == "__main__":
    main()