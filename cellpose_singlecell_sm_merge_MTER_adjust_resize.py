# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.exposure import adjust_log

import argparse

def merge_channels(args):
    ipdir = Path(args.i)
    sample_id = Path(args.id)
    chcolors = args.ch.split(' ')
    opimgpath = Path(args.o)
    opcsvpath = Path(args.c)
    img = []
    for chcolor in chcolors:
        img_tmp = imread(ipdir.joinpath(f'{sample_id}_{chcolor}.png'))
        img.append(img_tmp)
    img = np.stack(img, axis = -1)
    
    # merge yellow and red
    if len(chcolor) > 3:
       img[0] = img[0] + img[3]
       img[0] = np.clip(img[0], 0, 255)
       img = img[:, :, :3]
       
    img = adjust_log(img, 1)
    img = resize(img, (1024, 1024, 3), anti_aliasing=True)
    img = img * 255
    img = img.astype(np.uint8)
    
    opimgpath.parent.mkdir(exist_ok=True, parents=True)
    opcsvpath.parent.mkdir(exist_ok=True, parents=True)
    
    df = pd.DataFrame({  
                       'ipimgdir': [ipdir],
                       'opimgpath': [opimgpath],
                       'framesize_y': [img.shape[0]],
                       'framesize_x': [img.shape[1]]
                       })
    imsave(opimgpath, img)
    
    df.to_csv(opcsvpath, index=False)

    return

def main():
    description = "merge three channls into one single RGB file"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', type=str, required=True, help='input dir')
    parser.add_argument('-id', type=str, required=True, help='sample ID')
    parser.add_argument('-ch', type=str, default = 'red green blue yellow', help='channel order')
    parser.add_argument('-o', type=str, required=True, help='output img dir')
    parser.add_argument('-c', type=str, required=True, help='output csv dir')

    args = parser.parse_args()
    merge_channels(args) 
    return    

if __name__ == "__main__":
    main()