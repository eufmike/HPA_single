import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50


def get_filter_size(input_size, kernel_size):
    output_size = int((input_size - (kernel_size - 1)) / 2)
    return output_size


class simple_net(nn.Module):
    def __init__(self, input_ch, input_size, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv2d(input_ch, 6, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)

        filter_size = []
        for input_size_tmp in input_size:
            filter_size_tmp = get_filter_size(input_size_tmp, kernel_size)
            filter_size_tmp = get_filter_size(filter_size_tmp, kernel_size)
            filter_size.append(filter_size_tmp)
        self.fc1 = nn.Linear(16 * filter_size[0] * filter_size[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 19)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class custom_resnet(nn.Module):
    def __init__(self, input_ch, kernel_size=5):
        super(custom_resnet, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, 3, kernel_size, padding=2)
        self.conv2 = nn.Conv2d(3, 3, kernel_size, padding=2)
        self.resnet = resnet50()
        self.fc1 = nn.Linear(1000, 19)
        self.fc2 = nn.Linear(19, 19)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.resnet(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
