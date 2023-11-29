import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

default = transforms.Compose(
    [
    transforms.Resize((2048, 2048)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])  

v1_train = transforms.Compose(
    [
        A.Resize(height=1024, width=1024),
        A.Rotate(),
        A.HorizontalFlip(),
        A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
    )
v1_val = transforms.Compose(
    [

    ]
    )