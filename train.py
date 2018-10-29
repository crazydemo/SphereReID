#!/usr/bin/python
# -*- encoding: utf-8 -*-


import time
import logging
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backbone import Network_D
from sphere_loss import SphereLoss
from market1501 import Market1501
from balanced_sampler import BalancedSampler


## logging
FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


## TODO: use logger to show set up process
## TODO: warm up each iter or each epoch ?
def lr_scheduler(epoch, optimizer):
    warmup_epoch = 20
    warmup_lr = 5e-5
    start_lr = 1e-3
    lr_steps = [80, 100]
    lr_factor = 0.1

    if epoch < 20: # warmup
        warmup_scale = (start_lr / warmup_lr) ** (1.0 / 20)
        lr = warmup_lr * (warmup_scale ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for i, el in enumerate(lr_steps):
            if epoch == el:
                logger.info('LR is set to: {}'.format(lr))
                lr = start_lr * (lr_factor ** (i + 1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    return optimizer



def train():

    ## data
    dataset = Market1501('./dataset/Market-1501-v15.09.15/bounding_box_train')
    sampler = BalancedSampler(dataset, 16, 4)
    dl = DataLoader(dataset, batch_sampler = sampler, num_workers = 4)
    num_classes = dataset.get_num_classes()

    ## network and loss
    sphereloss = SphereLoss(1024, num_classes)
    sphereloss.cuda()
    net = Network_D()
    net = nn.DataParallel(net)
    net.train()
    net.cuda()

    ## optimizer
    params = list(net.parameters())
    params += list(sphereloss.parameters())
    optim = torch.optim.Adam(params, lr = 1e-3, weight_decay = 5e-4)


    ## training
    t_start = time.time()
    for ep in range(140):
        optim = lr_scheduler(ep, optim)
        for it, (imgs, lbs, ids) in enumerate(dl):
            imgs = imgs.cuda()
            lbs = lbs.cuda()
            embs = net(imgs)
            loss = sphereloss(embs, lbs)
            #  print(imgs.shape)
            #  print(embs.shape)
            #  print(lbs.shape)
            #  print(loss.detach().cpu().numpy())
            #  print(loss.shape)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if it % 10 == 0 and it != 0:
                t_end = time.time()
                log_loss = loss.detach().cpu().numpy()
                msg = 'epoch: {}, iter: {}, loss: {:4f}, time: {}'.format(ep,
                        it, log_loss, t_end - t_start)
                logger.info(msg)
                t_start = t_end


if __name__ == '__main__':
    train()