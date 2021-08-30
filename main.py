import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import kornia
import copy
import numpy
import argparse
from model import VGG

# --- Args
parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--pickup_from', type=int, default=0)
parser.add_argument('--save_prefix', type=str, default='')
parser.add_argument('--save_every', type=int, default=5)

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

init_model = lambda : VGG(n_channels=32)

train_dataset = CachedCIFAR10('../cl-pytorch/data', train=True)
test_dataset  = CachedCIFAR10('../cl-pytorch/data', train=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True)

kn_augs = nn.Sequential(
    kornia.augmentation.RandomCrop(size=(32,32), padding=4, fill=-1.9),
    kornia.augmentation.RandomHorizontalFlip(p=0.5)
).cuda()

model = init_model()

# --- Load the appropriate checkpoint
PATH = f'ckpt/epoch_{args.pickup_from}.pth'
print(f'starting from {PATH}')
model.load_state_dict(torch.load(PATH))

model.cuda()

print(model)
print(model[0][0].weight[0])

opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

best_model, best_acc = None, 0

for epoch in range(args.pickup_from, 150):
    # saving model here ensures `epochs` == number of epochs trained
    if epoch > 0 and epoch % args.save_every == 0:
        torch.save(model.state_dict(), f'ckpt/{args.save_prefix}epoch_{epoch}.pth')

    for (x,y) in train_loader:
        x, y = x.cuda(), y.cuda()
        x = kn_augs(x)

        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()

    if (epoch + 1) % 2 == 0:
        n_ok, n_total = 0, 0

        with torch.no_grad():
            for (x,y) in test_loader:
                x, y = x.cuda(), y.cuda()

                n_ok += model(x).argmax(1).eq(y).sum()
                n_total += x.size(0)

            acc = n_ok / n_total

            if acc > best_acc :
                best_acc = acc
                best_model = copy.deepcopy(model)

            print(f'epoch {epoch} acc : {acc:.4f}')


torch.save(best_model.state_dict(), f'final_model_augWS2.pth')
