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

# --- Args
parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--path_a', type=str)
parser.add_argument('--path_b', type=str)
args = parser.parse_args()

print(args)

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
    #model.train()

    n_ok, n_total, loss = 0, 0, 0
    for (x,y) in dl:
        x, y = x.cuda(), y.cuda()
        pred = model(x)

        n_ok += pred.argmax(1).eq(y).sum()
        n_total += y.size(0)

        loss += F.cross_entropy(pred, y, reduction='none').sum()

    return n_ok / n_total, loss / n_total


init_model = lambda : VGG(n_channels=32)

test_dataset = CachedCIFAR10('../cl-pytorch/data', train=False, download=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=256, drop_last=False, shuffle=True)

model_a = init_model()
model_b = init_model()

model_a.load_state_dict(torch.load(args.path_a))
model_b.load_state_dict(torch.load(args.path_b))

model_a.cuda(); model_b.cuda()

acc, loss = eval(model_a, test_dl)
print(f'Model A \t{acc:.4f}\t{loss:.4f}')

acc, loss = eval(model_b, test_dl)
print(f'Model B \t{acc:.4f}\t{loss:.4f}')

for alpha in (np.arange(10) / 10.):
    acc, loss = eval(model_interpolate(model_a, model_b, alpha=alpha), test_dl)
    print(f'interpolating with alpha {alpha}. Accuracy : {acc:.4f}\tLoss : {loss:.4f}')


'''
PATH = '/private/home/lucaspc/repos/crlapi/crlapi/sl/weight_files/'

# load models
model_task_a   = init_model()
model_task_a.load_state_dict(torch.load(PATH + 'VGG_n_channels:32_after0-2.pth'))

model_task_a_again   = init_model()
model_task_a_again.load_state_dict(torch.load(PATH + 'VGG_n_channels:32_after0-2-BOTH.pth'))


model_task_b   = init_model()
model_task_b.load_state_dict(torch.load(PATH + 'VGG_n_channels:32_after0-2-REV.pth'))

model_task_iid  = init_model()
model_task_iid.load_state_dict(torch.load(PATH + 'VGG_n_channels:32_after0-1.pth'))


out = model_interpolate(model_task_a, model_task_iid, alpha=0.1)

# do some interpolations
model_task_a.cuda()
acc, loss = eval(model_task_a, test_dl)
print(f'Task A \t{acc:.4f}\t{loss:.4f}')

model_task_b.cuda()
acc, loss = eval(model_task_b, test_dl)
print(f'Task B \t{acc:.4f}\t{loss:.4f}')

model_task_iid.cuda()
acc, loss = eval(model_task_iid, test_dl)
print(f'IID \t{acc:.4f}\t{loss:.4f}')

model_task_a_again.cuda()
acc, loss = eval(model_task_a_again, test_dl)
print(f'Task A again \t{acc:.4f}\t{loss:.4f}')

acc, loss = eval(model_interpolate(model_task_b, model_task_a_again, alpha=0.5), test_dl)
print(f'FROM A to A {acc:.4f}\t{loss:.4f}')

acc, loss = eval(model_interpolate(model_task_a, model_task_iid, alpha=0.5), test_dl)
print(f'interp 0.5 task a 0.5 task b {acc:.4f}\t{loss:.4f}')
'''


