# %%
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread, imsave

# %%
# import cellpoose
# combine files
prjdir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data")
datadir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification")
file_csv = datadir.joinpath("train_select_1K.csv")

# %%
df = pd.read_csv(file_csv)
print(df)

# %%
def load_img(ID, channels = [0, 1, 2, 3], total_ch = 4):
    channel_name_dict = {
        0: 'blue',
        1: 'green',
        2: 'red', 
        3: 'yellow'
    }
    if channels == 'all': 
        channels = list(range(total_ch))
    img = []
    for channel in channels:
        channel_name = channel_name_dict[channel]
        file_path = datadir.joinpath('train', ID + '_' + channel_name + '.png')
        img.append(imread(file_path))
    img = np.stack(img, axis = 0)
    return img

for i, row in df.iterrows():
    file_id = row['ID']
    file_path = datadir.joinpath('train', file_id)
    img = load_img(file_id)
    print(img.shape)
    
    break
    
# %%
