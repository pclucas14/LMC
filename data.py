import numpy

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F


# Data Prep
class CachedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)
        self.targets = numpy.array(self.targets)

        self.data = self.data / 255.

        # normalize
        mu  = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        var = torch.Tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

        self.data = (self.data - mu) / var


    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

class CachedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)
        self.targets = numpy.array(self.targets)

        self.data = self.data / 255.

        # normalize
        mu  = torch.Tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        var = torch.Tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

        self.data = (self.data - mu) / var


    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
