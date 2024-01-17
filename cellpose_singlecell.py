# %%
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread, imsave
from cellpose import models, plot, utils
import matplotlib.pyplot as plt

# %%
# import cellpoose
# combine files
prjdir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data")
datadir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification")
HPAdir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data")

# file_csv = datadir.joinpath("train_select_1K.csv")
file_csv = datadir.joinpath("train.csv")

# %%
df = pd.read_csv(file_csv)
print(df)

# %%
def load_img(ID, channels = [0, 1, 2, 3], total_ch = 4):
    channel_name_dict = {
        0: 'blue', # nucleus
        1: 'green', # protein
        2: 'red', # Microtubule
        3: 'yellow' # ER
    }
    if channels == 'all': 
        channels = list(range(total_ch))
    img = []
    for channel in channels:
        channel_name = channel_name_dict[channel]
        file_path = datadir.joinpath('train', ID + '_' + channel_name + '.png')
        img.append(imread(file_path))
    img = np.stack(img, axis = -1)
    return img

def img_display(img):
    fig, axarr = plt.subplots(1, img.shape[2], figsize = (20, 5))
    for i in range(img.shape[2]):
        axarr[i].imshow(img[:, :, i])
        axarr[i].axis('off')
    plt.show()
# %%
new_row = []
for i, row in tqdm(df[:10].iterrows()):
    file_id = row['ID']
    file_path = datadir.joinpath('train', file_id)
    
    channels_order = [2,1,0]
    img = load_img(file_id, channels=channels_order)
    
    oppath = HPAdir.joinpath('train_rgb_c210', file_id + '.png')
    oppath.parent.mkdir(parents=True, exist_ok=True)
    imsave(oppath, img)
    row['file_path'] = str(oppath)
    row['channel_order'] = ', '.join([str(x) for x in channels_order]) 
    row['channel_counts'] = len(channels_order)
    row['frame_size_y'] = img.shape[0]
    row['frame_size_x'] = img.shape[1]
    new_row.append(row)
    # img_display(img)

    # model = models.Cellpose(gpu = True, model_type = 'cyto2')
    # masks, flows, styles, diams = model.eval(img, diameter = 150, channels = [1, 3], flow_threshold = 0.4)
    # print(len(masks))
    # mask_RGB = plot.mask_overlay(img, masks)
    # outlines = utils.outlines_list(masks)
    # plt.imshow(img)
    # for o in outlines:
    #     plt.plot(o[:, 0], o[:, 1], 'r') 
    

new_row = pd.concat(new_row,ignore_index=True)
opcsvpath = datadir.joinpath('train_rgb_c210.csv')
new_row.to_csv(opcsvpath, index=False)
# %%

file_id = '1cb6bd56-bba5-11e8-b2ba-ac1f6b6435d0'
img = load_img(file_id, channels=[0, 1, 2, 3])
img_display(img)
# %%
