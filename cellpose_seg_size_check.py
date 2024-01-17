# %%
import os, sys
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from skimage.io import imread, imsave
from skimage.transform import resize

# %%
prjdir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data")
datadir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification")
HPAdir = Path("/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data")

# df_1K = pd.read_csv(datadir.joinpath('train_select_1K.csv'))
df = pd.read_csv(datadir.joinpath('train.csv'))
df_all_framesize = pd.read_csv(HPAdir.joinpath('all_merge.csv'))
df_all_framesize['ID'] = df_all_framesize['opimgpath'].apply(lambda x: Path(x).stem)    

df_merge = df.merge(df_all_framesize, how='left', on=['ID'])

# %%
display(df_merge.head())

print(len(df_merge))
# %%
df_gb = df_merge.groupby(['framesize_y', 'framesize_x']).size().reset_index(name='counts')
print(df_gb)

# %%
seed = 1947

df_sample = []
for i, row in df_gb.iterrows():
    df_tmp = df_merge[(df_merge['framesize_y'] == row['framesize_y']) & (df_merge['framesize_x'] == row['framesize_x'])]
    df_sample.append(df_tmp.sample(n=9, random_state=seed))
df_sample = pd.concat(df_sample, ignore_index=True)
print(df_sample.head())
# %%
from skimage.exposure import adjust_log
from tqdm import tqdm

opdir = HPAdir.joinpath('sample_merge')
for i, row in tqdm(df_sample.iterrows()):
    oppath = opdir.joinpath(f"{row['framesize_y']}_{row['framesize_x']}", f"{row['ID']}.png") 
    ippath = row['opimgpath']
    oppath.parent.mkdir(exist_ok=True, parents=True)
    # shutil.copy2(ippath, oppath)

    img = imread(ippath)
    img = adjust_log(img, 1)
    img = resize(img, (1024, 1024, 3), anti_aliasing=True)
    img = img * 255
    img = img.astype(np.uint8)
    imsave(oppath, img)
# %%
