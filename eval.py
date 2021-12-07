import torch
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import copy
import numpy
import numpy as np
import argparse
from model import VGG
from data import CachedCIFAR10

def get_args():
    # --- Args
    parser = argparse.ArgumentParser()

    """ optimization (fixed across all settings) """
    parser.add_argument('--path_a', type=str)
    parser.add_argument('--path_b', type=str)
    args = parser.parse_args()
    return args


def model_interpolate(model_a, model_b, alpha):
    new_model = copy.deepcopy(model_a)
    new_model.to(next(iter(model_a.parameters())).device)

    a_dict = model_a.state_dict()
    b_dict = model_b.state_dict()

    for key in a_dict.keys():
        pa, pb = a_dict[key], b_dict[key]
        new_model.state_dict()[key].data.copy_(pa.data * alpha + pb.data * (1 - alpha))

    return new_model


@torch.no_grad()
def eval(model, dl):
    model.eval()

    n_ok, n_total, loss = 0, 0, 0
    for (x,y) in dl:
        x, y = x.cuda(), y.cuda()
        pred = model(x)

        n_ok += pred.argmax(1).eq(y).sum().item()
        n_total += y.size(0)

        loss += F.cross_entropy(pred, y, reduction='none').sum()

    return n_ok / n_total, loss / n_total


if __name__ == '__main__':
    args = get_args()

    init_model = lambda : VGG(n_channels=32)

    test_dataset = CachedCIFAR10('../cl-pytorch/data', train=False, download=True)
    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=True
    )

    model_a = init_model()
    model_b = init_model()

    model_a.load_state_dict(torch.load(args.path_a))
    model_b.load_state_dict(torch.load(args.path_b))

    model_a.cuda(); model_b.cuda()

    acc, loss = eval(model_a, test_dl)
    print(f'Model A \t{acc:.4f}\t{loss:.4f}')

    acc, loss = eval(model_b, test_dl)
    print(f'Model B \t{acc:.4f}\t{loss:.4f}')

    for alpha in (np.arange(11) / 10.):
        acc, loss = eval(model_interpolate(model_a, model_b, alpha=alpha), test_dl)
        print(f'interpolating with alpha {alpha}. Accuracy : {acc:.4f}\tLoss : {loss:.4f}')

