import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2

import albumentations as A
from albumentations.pytorch import ToTensorV2


tf_dataset = transforms.Compose(
        [
            transforms.ToTensor(),
            v2.Resize((1024, 1024)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ]
    )  

default = transforms.Compose(
        [
            transforms.Resize(size = (1024, 1024), antialias = True),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ]
    )  

'''
tv_v1_train = transforms.Compose(
        [   
            v2.Resize(size = (1024, 1024), antialias = True)
            # v2.RandomResizedCrop(size=(224, 224), antialias=True),
            # v2.RandomHorizontalFlip(p = 0.5),
            # v2.RandomVerticalFlip(p = 0.5),
            # v2.RandomRotation(degrees=(0, 180)),
            # v2.ToDtype(torch.float32),
            # v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
        ]
    )
'''