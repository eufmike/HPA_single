# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread

# %%
imgpath = Path(
    "/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification/train/0a00aab2-bbbb-11e8-b2ba-ac1f6b6435d0_blue.png"
)
img = imread(imgpath)
print(img.shape)
print(img.dtype)

# %%
a = np.arrary([1, 2, 3])
print(a)
