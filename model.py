import torch
import torch.nn as nn
import numpy as np

# -- Layers

def _make_layers(array, in_channels):
    layers = []
    for x in array:

        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x

    return in_channels, nn.Sequential(*layers)


def VGG(n_channels=-1):
    vgg_parts = [ [64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 256, 'M'], [512, 512, 512, 512, 'M'], [512, 512, 512, 512, 'M'] ]

    if n_channels > 0:
        vgg_parts = [[n_channels if type(x) == int else x for x in block] for block in vgg_parts]

    in_channels, block0   = _make_layers(vgg_parts[0], 3)
    in_channels, block1 = _make_layers(vgg_parts[1], in_channels)
    in_channels, block2 = _make_layers(vgg_parts[2], in_channels)
    in_channels, block3 = _make_layers(vgg_parts[3], in_channels)
    in_channels, block4 = _make_layers(vgg_parts[4], in_channels)

    return nn.Sequential(
        block0,
        block1,
        block2,
        block3,
        block4,
        nn.Flatten(),
        nn.Linear(in_channels, 10)
    )

