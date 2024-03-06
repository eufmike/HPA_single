# %%
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

'''
for filepath in filelist[:5]:
    img = imread(filepath)
    model = models.Cellpose(gpu = True, model_type = 'cyto2')
    masks, flows, styles, diams = model.eval(img, 
                      diameter = 150, 
                      channels = [1, 3], 
                      flow_threshold = 0.8,
                      cellprob_threshold = 0.0
                      )
    opdatapath = datapath.joinpath(f'{filepath.stem}.npy')
    opsegpath = segpath.joinpath(f'{filepath.stem}.png')
    io.masks_flows_to_seg(img, masks, flows, diams, opdatapath)
    io.save_to_png(img, masks, flows, opsegpath)
'''

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
