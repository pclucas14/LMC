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

# --- Args
parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--init_path', type=str, default=None) #'/checkpoint/lucaspc/LMC/cifar/ckpt/epoch_0.pth')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_prefix', type=str, default='')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--replay', type=int, default=0)
parser.add_argument('--n_tasks', type=int, default=10)
parser.add_argument('--start_task', type=int, default=0)
parser.add_argument('--end_task', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--task_order', type=int, nargs='+', default=None)
parser.add_argument('--init_from_best', type=int, default=1)
parser.add_argument('--delta', type=float, default=0.)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--keep_k_hardest', type=float, default=-1)
parser.add_argument('--keep_k', type=int, default=-1)
parser.add_argument('--loss', type=str, default='ce', choices=['supcon', 'ce'])

args = parser.parse_args()
if args.task_order is None:
    args.task_order = numpy.arange(args.n_tasks)

print(args)


wandb.init(project='alma_simple', name=args.exp_name, config=args)

if args.model == 'vgg':
    init_model = lambda : VGG(n_channels=32)
else:
    init_model = lambda : resnet18(num_classes=10) #VGG(n_channels=32)

train_dataset = CachedCIFAR10('../cl-pytorch/data', train=True)
test_dataset  = CachedCIFAR10('../cl-pytorch/data', train=False)
test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

kn_augs = nn.Sequential(
    kornia.augmentation.RandomCrop(size=(32,32), padding=4, fill=-1.9),
    kornia.augmentation.RandomHorizontalFlip(p=0.5)
).cuda()

model = init_model()
model.cuda()

# --- Load the appropriate checkpoint
PATH = args.init_path
print(f'starting from {PATH}')
try:
    model.load_state_dict(torch.load(PATH))
    print(model[0][0].weight[0])
except:
    print('ERROR LOADING WEIGHTS')

print(model)
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


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

# --- Loss init
if args.loss == 'supcon':
    loss_fn = SupCon()
    protos = model.linear.weight
    model.linear = nn.Identity()

    b_proto = torch.zeros_like(protos)
    b_count = torch.zeros(size=(protos.size(0),)).to(protos.device).long()
    arr_D   = torch.arange(protos.size(1)).to(protos.device)

    @torch.no_grad()
    def update_protos(protos, features, y, M=0.99):
        global b_proto, b_count, arr_D

        if torch.isnan(features).any():
            import pdb; pdb.set_trace()
            xx =1

        b_proto.fill_(0)
        b_count.fill_(0)

        features = F.normalize(features, dim=-1, p=2)
        protos   = F.normalize(protos, dim=-1, p=2)

        out_idx = arr_D.view(1, -1) + y.view(-1, 1) * protos.size(-1)
        b_proto = b_proto.view(-1).scatter_add(0, out_idx.view(-1), features.view(-1)).view_as(b_proto)
        b_count.scatter_add_(0, y, torch.ones_like(y))

        # do the actual update
        b_proto[b_count == 0] = protos[b_count == 0]

        new_protos = protos * M + b_proto * (1 - M)

        return new_protos

    def predict(hid):
        global protos

        # logits : bs X D, protos : C x D
        hid    = F.normalize(hid, dim=-1, p=2)
        protos = F.normalize(protos, dim=-1, p=2)

        dist = (protos.unsqueeze(0) - hid.unsqueeze(1)).pow(2).sum(-1)
        return dist.argmin(1)

    loss_fn = SupCon()
else:
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    def predict(logits):
        return logits.argmax(1)

# ---- LOOP over Mega-batches
step = 0
seen_tasks = []
n_epochs = args.epochs

for task in args.task_order:

    if len(seen_tasks) < args.start_task:
        print(f'skipping task {task}')
        seen_tasks += [task]
        continue

    seen_tasks += [task]

    if args.replay:
        task_train_ds = torch.utils.data.ConcatDataset([train_chunks[i] for i in seen_tasks])
        task_val_ds   = torch.utils.data.ConcatDataset([val_chunks[i] for i in seen_tasks])
    else:
        task_train_ds = train_chunks[task]
        task_val_ds   = val_chunks[task]

    S = True
    tr_loader  = torch.utils.data.DataLoader(task_train_ds, batch_size=args.batch_size, shuffle=S, drop_last=True)
    val_loader = torch.utils.data.DataLoader(task_val_ds, batch_size=1024, shuffle=S)

    print(f'task {task}:\t{len(task_train_ds)} training & {len(task_val_ds)} val samples')

    best_opt    = opt
    best_model  = model
    best_acc, _ = eval(best_model, val_loader, predict, loss_fn)

    for epoch in range(n_epochs):

        model.train()

        if epoch == 0:
            loss, acc = get_stats(task_train_ds, model, loss_fn=loss_fn)
            acc  = acc.float().mean().item()
            np_hist = np.histogram(loss.cpu().numpy())
            hist = wandb.Histogram(np_histogram=np_hist)
            wandb.log({'train/loss_hist': hist, 'train/init_acc': acc}, step=step)

            if (args.keep_k_hardest > 0 or args.keep_k > 0) and task > 0:
                tr_loader = torch.utils.data.DataLoader(
                        task_train_ds,
                        batch_size=args.batch_size,
                        sampler=WeightedSampler(
                            dataset=task_train_ds,
                            loss=loss.cpu().numpy(),
                            qt=args.keep_k_hardest,
                            size=args.keep_k
                        ),
                        drop_last=True
                )

        start = time.time()
        train_loss = 0

        for it, (x,y) in enumerate(tr_loader):
            x, y = x.cuda(), y.cuda()
            x = kn_augs(x)

            opt.zero_grad()
            output = model(x)
            loss = loss_fn(output, y).mean()
            loss.backward()
            opt.step()
            step += 1

            if args.loss == 'supcon':
                protos = update_protos(protos, output, y)

            train_loss += loss.item()

        epoch_time = time.time() - start

        acc, loss = eval(model, val_loader, predict, loss_fn)

        if acc > (best_acc + args.delta):
            best_acc = acc
            best_model = copy.deepcopy(model)
            best_opt = copy.deepcopy(opt)

        wandb.log({
            'task': task,
            'train/loss':train_loss / (it+1),
            'train/time': epoch_time,
            'valid/acc': acc,
            'valid/loss': loss}, step=step
        )

        print(f'epoch {epoch} acc : {acc:.4f}\t{epoch_time:.2f}')

    if task == args.end_task and args.end_task > 0:
        print(f'stopping after having seen {seen_tasks}')
        exit()

    # Start next task with best model
    if args.init_from_best:
        opt.load_state_dict(best_opt.state_dict())
        for pa, pb in zip(model.state_dict().values(), best_model.state_dict().values()):
            pa.data.copy_(pb.data)

    acc, loss = eval(model, test_loader, predict, loss_fn)
    print(f'test acc {acc:.4f}')
    wandb.log({'test/acc': acc, 'test/loss': loss}, step=step)

acc, loss = eval(model, test_loader, predict, loss_fn)
print(f'test acc {acc:.4f}')
