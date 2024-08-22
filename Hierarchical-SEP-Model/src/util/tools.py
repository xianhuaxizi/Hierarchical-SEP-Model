# coding=utf-8
import numpy as np
import json
import math
import os
import pickle
import re
import sys
import time
import configparser  # https://www.cnblogs.com/dion-90/p/7978081.html 读写ini
from scipy import optimize
from sklearn import preprocessing

#nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module, Parameter, init
from torch.optim import lr_scheduler

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def list_to_torch(list):
    tensor = to_torch(np.array(list)).float()
    return tensor

def get_word_embedding(root_path):
    try:
        return np.load(root_path+"word_embedding.npy")
    except FileNotFoundError:
        with open(root_path+"deepwalk_128_unweighted_with_args.txt") as f:
            index_embedding = {}
            for line in f:
                line = line.strip().split()
                if len(line) == 2:
                    continue
                index_embedding[line[0]] = np.array(line[1:], dtype=np.float32)
            index_embedding["0"] = np.zeros(len(index_embedding["0"]), dtype=np.float32)
        word_embedding = []
        for i in range(len(index_embedding)):
            word_embedding.append(index_embedding[str(i)])
        word_embedding = np.array(word_embedding, dtype=np.float32)
        np.save(root_path+"word_embedding", word_embedding)
        return word_embedding


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, step, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if schedule and (step % schedule == 0):
        lr *= gamma
       # lr = max(lr, 1e-7)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay, threshold=0.01, patience=opt.patients)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'warmup_constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: math.log(epoch + 1)/math.log(opt.warm_up_epochs) if epoch < opt.warm_up_epochs else 1)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / opt.warm_up_epochs if epoch <= opt.warm_up_epochs else 1)
    elif opt.lr_policy == 'constant':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler