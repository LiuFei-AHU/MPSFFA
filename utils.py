from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import math
import pandas as pd


class DiscriminateLoss(Module):
    def __init__(self):
        super(DiscriminateLoss, self).__init__()

    def forward(self, input, label):
        if label.shape[0] == 2:
            mse = torch.sqrt(torch.square(torch.subtract(input[0], input[1])))
            mmse = torch.mean(mse)
            if label[0] == label[1]:
                return -torch.log(1 - mmse)
            else:
                return torch.exp(-mmse)
                # return 1.0/mmse
        return torch.tensor(0)


class CosineSimilarityLoss(Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.flatten = torch.nn.Flatten(2)

    def forward(self, input, label):
        input = self.flatten(input)
        if label.shape[0] == 2:
            d = torch.sqrt(torch.square(torch.nn.functional.cosine_similarity(input[0], input[1])))
            if label[0] != label[1]:
                return -1 * torch.mean(1.0 / torch.log(d))
            else:
                return -1 * torch.mean(torch.log(d))
        return 0


def image_aug(img, augs):
    for aug in augs:
        img = aug(img)
    return torch.tensor(img, dtype=torch.float32)


def get_warm_rate(input):  # input
    warm_step = 5
    down_k = 0.5

    if not (hasattr(get_warm_rate, 'indx')):
        get_warm_rate.indx = 1

    if input == 0:
        get_warm_rate.indx = 1

    if input < warm_step:
        lr_k = (input + 1) * (1 / warm_step)

    elif input <= warm_step + 5:
        lr_k = 1
    else:
        lr_k = down_k ** get_warm_rate.indx
        get_warm_rate.indx += 1

    return lr_k


def is_upadte_lr(val_loss, epoch):
    if not (hasattr(is_upadte_lr, 'loss')):
        is_upadte_lr.loss = []
    if not (hasattr(is_upadte_lr, 'flag')):
        is_upadte_lr.flag = 0
    if not (hasattr(is_upadte_lr, 'cool')):
        is_upadte_lr.cool = 0

    if epoch == 0:
        is_upadte_lr.loss = []
        is_upadte_lr.flag = 0
        is_upadte_lr.cool = 0

    if len(is_upadte_lr.loss) > 3:
        mean = np.mean(is_upadte_lr.loss)
        if val_loss > mean:
            is_upadte_lr.flag += 1
        del is_upadte_lr.loss[0]
    is_upadte_lr.loss.append(val_loss)

    is_upadte_lr.cool += 1
    if is_upadte_lr.flag > 2 and is_upadte_lr.cool > 3:
        is_upadte_lr.flag = 0
        is_upadte_lr.cool = 0
        return True
    else:
        return False


def get_new_lr(epoch, train_loss, val_loss, lr):
    if not (hasattr(get_new_lr, 'loss')):
        get_new_lr.loss = val_loss
    if epoch == 0:
        return lr
    if val_loss > train_loss:
        lr = lr * (val_loss // train_loss)
    elif abs(get_new_lr.loss - val_loss) <= val_loss * 0.2:
        lr = lr * 2

    return max(2e-6, min(lr, 1e-4))


class EarlyStopping(object):
    def __init__(self, patience=3, min_delta=0.01, best_acc=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = best_acc
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc > self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        elif val_acc - self.best_acc <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
        return self.early_stop


class StepDownCosDecay(object):
    """
    Customized Learning rate decay strategy. Combine the StepLR and CosineAnnealingLR.
    """

    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=0,
                 last_epoch=-1,
                 endure_epochs=1,
                 verbose=False,
                 gamma=0.8):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingLR' must be 'int', but received %s."
                % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingLR' must be 'float, int', but received %s."
                % type(eta_min))
        assert T_max > 0 and isinstance(
            T_max, int), " 'T_max' must be a positive integer."
        self.last_epoch = last_epoch
        self.base_lr = learning_rate
        self.last_lr = self.base_lr
        self.T_max = T_max
        self.eta_min = float(eta_min)
        self.gamma = gamma
        self.verbose = verbose
        self.endure_epochs = endure_epochs
        self.print()

    def step(self):
        self.last_epoch += 1
        return self.get_lr()

    def print(self):
        if self.verbose:
            print(f'StepDownCosDecay set learning rate to {self.last_lr}')

    def get_lr(self):
        if self.last_epoch > self.endure_epochs:
            self.last_lr = self.get_cos_lr() * math.pow(self.gamma, self.last_epoch // self.T_max)
        self.print()
        return self.last_lr

    def get_cos_lr(self):
        if self.last_epoch <= 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max)) / 2


class LogisticLoss(Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.sqrt(torch.square(torch.subtract(x, y))))


class LogcoshLoss(Module):
    def __init__(self):
        super(LogcoshLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.log(torch.cosh(torch.subtract(y, x))))


def get_pmci_ids(csv_path):
    pmcis = []
    for p in csv_path:
        df = pd.read_csv(p)
        pmcis.extend([k[0] for k in df[['Subject ID']].values.tolist()])
    return pmcis
