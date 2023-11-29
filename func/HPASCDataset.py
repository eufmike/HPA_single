import os, sys
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from skimage.io import imread, imsave

class TrDataset(Dataset):
  def __init__(self, base_dataset, transform):
    super(TrDataset, self).__init__()
    self.base = base_dataset
    self.transform = transform

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y = self.base[idx]
    return self.transform(x), y

class HPASCDataset(Dataset):
    def __init__(self, input_csv, root, transform = None, input_ch_ct = 'all', n_class = 19, debug_size = None):

        self.transform = transform
        df_input = pd.read_csv(input_csv)
        df_input = df_input[:debug_size]
        self.imgs_stem = df_input['ID'].apply(lambda x: Path(root).joinpath(x))
        self.lbls = df_input['Label'] 
        self.channels = ['blue', 'green', 'red', 'yellow']
        
        if isinstance(input_ch_ct, int): 
            self.channels = self.channels[:input_ch_ct]
        elif isinstance(input_ch_ct, list): 
            self.channels = [self.channels[ch] for ch in input_ch_ct]
        elif input_ch_ct == 'all': 
            pass
            
        self.n_class = n_class
        
        assert len(self.imgs_stem) == len(self.lbls), 'mismatched length!'

    def __getitem__(self, index):
        # print(index)
        imgpath_noext = self.imgs_stem[index]
        img = []
        # print(self.channels)
        for ch in self.channels:
            imgpath_tmp = Path(f'{imgpath_noext}_{ch}.png')
            # print(imgpath_tmp)
            # img_tmp = Image.open(imgpath_tmp).convert('gray')
            img_tmp = Image.open(imgpath_tmp)
            img_tmp = img_tmp.resize((2048, 2048))
            img_tmp = np.array(img_tmp)
            img.append(img_tmp)
        img = np.stack(img, axis = 0)
        img = img.astype('float32') / 255.0
        # print(img.shape)
        
        lbl_str = self.lbls[index]
        lbl = np.zeros(self.n_class, dtype='float32')
        for lbl_tmp in lbl_str.split('|'):
            lbl_tmp = int(lbl_tmp)
            assert lbl_tmp < self.n_class, 'label out of range!'
            lbl[lbl_tmp] = 1
        # print(lbl)

        if not self.transform is None:
            img = self.transform(img)

        return img, lbl
        
    def __len__(self):
        return len(self.imgs_stem)
    
if __name__ == '__main__':
    pass