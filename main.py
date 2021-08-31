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
from model import VGG
from data import CachedCIFAR10
from eval import eval

# --- Args
parser = argparse.ArgumentParser()

""" optimization (fixed across all settings) """
parser.add_argument('--init_path', type=str, default='/checkpoint/lucaspc/LMC/cifar/ckpt/epoch_0.pth')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--save_prefix', type=str, default='')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--replay', type=int, default=0)
parser.add_argument('--n_tasks', type=int, default=1)
parser.add_argument('--start_task', type=int, default=0)
parser.add_argument('--end_task', type=int, default=-1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--task_order', type=int, nargs='+', default=None)
parser.add_argument('--init_from_best', type=int, default=1)
parser.add_argument('--delta', type=float, default=0.)
parser.add_argument('--exp_name', type=str, default=None)

args = parser.parse_args()
if args.task_order is None:
    args.task_order = numpy.arange(args.n_tasks)

print(args)


wandb.init(project='LMC', name=args.exp_name, config=args)

init_model = lambda : VGG(n_channels=32)

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
model.load_state_dict(torch.load(PATH))
print(model[0][0].weight[0])

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



# ---- LOOP over Mega-batches
seen_tasks = []

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
    tr_loader  = torch.utils.data.DataLoader(task_train_ds, batch_size=args.batch_size, shuffle=S)
    val_loader = torch.utils.data.DataLoader(task_val_ds, batch_size=1024, shuffle=S)

    print(f'task {task}:\t{len(task_train_ds)} training & {len(task_val_ds)} val samples')

    best_model  = model
    best_acc, _ = eval(best_model, val_loader)

    for epoch in range(args.epochs):

        model.train()
        start = time.time()
        train_loss = 0

        for it, (x,y) in enumerate(tr_loader):
            x, y = x.cuda(), y.cuda()
            x = kn_augs(x)

            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        epoch_time = time.time() - start

        acc, loss = eval(model, val_loader)

        if acc > (best_acc + args.delta):
            best_acc = acc
            best_model = copy.deepcopy(model)

        wandb.log({
            'task': task,
            'train/loss':train_loss / (it+1),
            'train/time': epoch_time,
            'valid/acc': acc,
            'valid/loss': loss}, step=args.epochs * task + epoch
        )

        print(f'epoch {epoch} acc : {acc:.4f}\t{epoch_time:.2f}')

    if (task + 1) % args.save_every == 0:
        print('saving')
        torch.save(best_model.state_dict(), f'ckpt/{args.save_prefix}_best_t{task}.pth')
        torch.save(model.state_dict(), f'ckpt/{args.save_prefix}_t{task}.pth')

    if task == args.end_task and args.end_task > 0:
        print(f'stopping after having seen {seen_tasks}')
        exit()

    # Start next task with best model
    if args.init_from_best:
        for pa, pb in zip(model.state_dict().values(), best_model.state_dict().values()):
            pa.data.copy_(pb.data)

    acc, loss = eval(model, test_loader)
    print(f'test acc {acc:.4f}')
    wandb.log({'test/acc': acc, 'test/loss': loss}, step=args.epochs * task + epoch)

acc, loss = eval(model, test_loader)
print(f'test acc {acc:.4f}')
