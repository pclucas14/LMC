import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections.abc import Iterable

@torch.no_grad()
def get_stats(loader, model, loss_fn=lambda x, y: F.cross_entropy(x, y, reduction='none')):
    model.eval()

    loader = torch.utils.data.DataLoader(loader, shuffle=False, batch_size=1024)

    losses = []
    accs   = []

    for (x, y) in loader:
        x, y = x.cuda(), y.cuda()
        out = model(x)

        loss = loss_fn(out, y)
        acc  = out.argmax(1).eq(y)

        losses += [loss]
        accs += [acc]

    model.train()

    return torch.cat(losses), torch.cat(accs)


class WeightedSampler(torch.utils.data.Sampler):
    """ Upsample points based on sample weight """

    def __init__(self, dataset, loss, qt, size):
        if qt != -1:
            th = np.quantile(loss * -1, qt) # we want the biggest valus
            self.len = len(dataset)
            self.idxs = np.argwhere(loss * -1 < th)[:, 0]
        elif size != -1:
            assert qt == -1
            assert size <= len(dataset)
            self.idxs = np.argsort(loss)[-size:]
            self.len = size

    def __iter__(self):
        cnt = 0
        while cnt < self.len:
            cnt += 1
            # sample random item
            yield np.random.choice(self.idxs)

    def __len__(self):
        return self.len
