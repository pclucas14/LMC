import torch
import torch.nn as nn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F

import wandb
import time
import copy
import numpy
import kornia
import argparse
from model import *
from data import CachedCIFAR10
from eval import eval
from utils import get_stats, WeightedSampler
from losses import SupCon

def get_args_parser():
    # --- Args
    parser = argparse.ArgumentParser(add_help=False)

    """ optimization (fixed across all settings) """
    parser.add_argument('--init_path', type=str, default='/private/home/lucaspc/repos/LMC/testbed/og_weights.pth')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--replay', type=int, default=0)
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--start_task', type=int, default=0)
    parser.add_argument('--end_task', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init_from_best', type=int, default=1)
    parser.add_argument('--delta', type=float, default=0.)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--keep_k_hardest', type=float, default=-1)
    parser.add_argument('--keep_k', type=int, default=-1)
    parser.add_argument('--loss', type=str, default='ce', choices=['supcon', 'ce'])

    return parser

def train_model(args):

    wandb.init(project='alma_simple', name=args.exp_name, config=args)

    if args.model == 'vgg':
        init_model = lambda : VGG(n_channels=32)
    else:
        init_model = lambda : resnet18(num_classes=10) #VGG(n_channels=32)

    train_dataset = CachedCIFAR10('/private/home/lucaspc/repos/cl-pytorch/data', train=True)
    test_dataset  = CachedCIFAR10('/private/home/lucaspc/repos/cl-pytorch/data', train=False)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    kn_augs = nn.Sequential(
        kornia.augmentation.RandomCrop(size=(32,32), padding=4, fill=-1.9),
        kornia.augmentation.RandomHorizontalFlip(p=0.5)
    ).cuda()

    modela = init_model()
    modela.cuda()

    modelb = init_model()
    modelb.cuda()

    modelboth = init_model()
    modelboth.cuda()

    modela.load_state_dict(torch.load('./testbed/set_0.pth'))
    modelb.load_state_dict(torch.load('./testbed/set_1.pth'))
    modelboth.load_state_dict(torch.load('./testbed/set_both_2.pth'))

    models = {'a' : modela, 'b' : modelb, 'ab': modelboth}

    # --- Split Data into tasks
    # MAKE SURE ALWAYS SAME SPLIT
    total_size = len(train_dataset)
    chunks = [total_size // args.n_tasks] * args.n_tasks
    chunks = torch.utils.data.random_split(train_dataset, chunks, generator=torch.Generator().manual_seed(42))
    train_chunks, val_chunks = [], []

    for chunk in chunks:
        size     = len(chunk)
        tr_size  = int(size * 0.9)
        val_size = size - tr_size

        tr_chunk, val_chunk = torch.utils.data.random_split(chunk, [tr_size, val_size], generator=torch.Generator().manual_seed(42))
        train_chunks += [tr_chunk]
        val_chunks   += [val_chunk]

    @torch.no_grad()
    def get_all_preds(model, loader):

        preds, ys = [], []
        for i, (x,y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            pred = model(x).argmax(1)

            preds += [pred]
            ys += [y]

        return torch.cat(preds), torch.cat(ys)

    res = {}
    # Things to look at :
    for task in range(2):
        tr_loader = torch.utils.data.DataLoader(train_chunks[task], batch_size=512)
        val_loader = torch.utils.data.DataLoader(val_chunks[task], batch_size=512)

        for name, model in models.items():
            tr_p, tr_y = get_all_preds(model, tr_loader)
            te_p, te_y = get_all_preds(model, val_loader)

            res[f'{name}_{task}_tr'] = (tr_p, tr_y)
            res[f'{name}_{task}_te'] = (te_p, te_y)

            tr_acc = (tr_p == tr_y).float().mean()
            te_acc = (te_p == te_y).float().mean()

            print(f'{name}\t {task}\t tr acc {tr_acc:.3f}\t te acc {te_acc:.3f}')

    for task in range(2):
        print(f'task {task}')
        # where did a mess up
        pred_a, tgt_a = res[f'a_{task}_te']
        a_err_idx = torch.where( pred_a != tgt_a )[0]

        pred_b, tgt_b = res[f'b_{task}_te']
        b_err_idx = torch.where( pred_b != tgt_b )[0]

        pred_ab, tgt_ab = res[f'ab_{task}_te']

        b_got_ok = (pred_b == tgt_b)[a_err_idx].float().mean()
        ab_got_ok = (pred_ab == tgt_ab)[a_err_idx].float().mean()
        print(f'over a\'s {a_err_idx.size(0)} mistakes, b got {b_got_ok.item():.2f}, ab got {ab_got_ok.item():.2f}')

        a_got_ok = (pred_b == tgt_b)[b_err_idx].float().mean()
        ab_got_ok = (pred_ab == tgt_ab)[b_err_idx].float().mean()
        print(f'over b\'s {b_err_idx.size(0)} mistakes, a got {b_got_ok.item():.2f}, ab got {ab_got_ok.item():.2f}')

        assert not (tgt_a != tgt_b).any()

        print(f'when a & b agree, they get {(pred_a == tgt_a)[pred_a == pred_b].float().mean():.2f} vs {(pred_ab == tgt_ab)[pred_a == pred_b].float().mean():.2f} for ab')

        diff = torch.where(pred_a != pred_b)[0]
        a_ok = (pred_a == tgt_a)[diff].float().mean()
        b_ok = (pred_b == tgt_b)[diff].float().mean()
        either = ((pred_a == tgt_a) | (pred_b == tgt_b))[diff].float().mean()

        print(f'when disagree, a gets {a_ok:.2f}, b gets {b_ok:.2f}, either {either:.2f}')


    import pdb; pdb.set_trace()
    xx = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LMC', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)

