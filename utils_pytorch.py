import numpy as np
import torch
from torch.nn import Module
from torch.optim import lr_scheduler
import math
import pandas as pd


class EarlyStopping():
    def __init__(self, patience=3, min_delta=0.01, best_acc=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = best_acc
        self.early_stop = False
    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc> self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        elif val_acc - self.best_acc <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
        return self.early_stop


# Customized learning rate
class StepDownCosDecay(lr_scheduler):
    def __init__(self,
                learning_rate,
                T_max,
                eta_min=0,
                last_epoch=-1,
                verbose=False,
                gamma=0.8):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        assert T_max > 0 and isinstance(
            T_max, int), " 'T_max' must be a positive integer."
        self.T_max = T_max
        self.eta_min = float(eta_min)
        self.gamma=gamma
        super(StepDownCosDecay, self).__init__(learning_rate, last_epoch,verbose)

    def get_lr(self):
        return self.get_cos_lr() * math.pow(self.gamma, self.last_epoch//self.T_max)

    def get_cos_lr(self):
        if self.last_epoch == 0:
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


class DiscriminateLoss(Module):
    def __init__(self):
        super(DiscriminateLoss, self).__init__()
        self.flatten = torch.nn.Flatten(2)

    def forward(self, input, label):
        input = self.flatten(input)
        if label.shape[0] == 2:
            d = torch.sqrt(torch.square(torch.nn.functional.cosine_similarity(input[0], input[1])))
            if label[0] != label[1]:
                return -torch.mean(1.0/torch.log(d))
            else:
                return -torch.mean(torch.log(d))
        return 0


def get_pmci_ids(csv_path):
    pmcis = []
    for p in csv_path:
        df = pd.read_csv(p)
        pmcis.extend([k[0] for k in df[['Subject ID']].values.tolist()])
    return pmcis


def train_mmse(model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop, save_model_path,
               last_best_acc=0.5, weight=None, t_rate=1):
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    best_acc = last_best_acc
    best_acc_actual = 0.0
    min_loss = 100.0
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []

    print('start train ..')
    for epoch_id in range(EPOCH_NUM):
        model.train()
        los_list = []
        acc_list = []
        for batch_id, data in enumerate(train_loader()):
            image, label, mmse = data

            predict = model(image, mmse)
            loss = loss_fn(predict, label)

            acc = torch.metric.accuracy(predict, label)
            acc_list.extend(acc.numpy())
            los_list.append(loss.numpy())

            loss_str = "epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy())
            print(loss_str)
            logger.write(loss_str + '\n')

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id >= t_rate * len(train_loader):
                break

        epoch_loss = np.mean(los_list)
        epoch_acc = np.mean(acc_list)
        epoch_loss_str = "epoch: {}, acc is {} ,loss is: {}".format(epoch_id, epoch_acc, epoch_loss)
        print(epoch_loss_str)
        logger.write(epoch_loss_str + '\n')
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        model.eval()
        acc_list = []
        los_list = []
        print('testing ...')
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader()):
                x_data, y_data, mmse = data
                predicts = model(x_data, mmse)
                loss = loss_fn(predicts, y_data)
                acc = torch.metric.accuracy(predicts, y_data)
                acc_list.extend(acc.numpy())
                los_list.append(loss.numpy())
                eval_loss_str = "batch_id: {}, acc is: {}".format(batch_id, acc.numpy())
                print(eval_loss_str)
                logger.write(eval_loss_str + '\n')

        val_acc = np.mean(acc_list)
        val_los = np.mean(los_list)
        epoch_eval_loss_str = 'val acc is {} , val loss is {}'.format(val_acc, val_los)
        print(epoch_eval_loss_str)
        logger.write(epoch_eval_loss_str + '\n')
        eval_loss.append(val_los)
        eval_acc.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
        elif val_acc == best_acc and val_los < min_loss:
            min_loss = val_los
            torch.save(model.state_dict(), save_model_path)
        if val_acc >= best_acc_actual:
            best_acc_actual = val_acc
        scheduler.step()

        if early_stop is not None and early_stop(val_acc):
            break

    logger.write('best acc is {} \n'.format(best_acc_actual))
    if not logger.closed:
        logger.close()

    return train_loss, eval_loss, train_acc, eval_acc


def train_mmse_similarity(model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop,
                          save_model_path, last_best_acc=0.5, similarity_loss=None, loss_rate=0.01, weight=None,
                          t_rate=1):
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    best_acc = last_best_acc
    best_acc_actual = 0.0
    min_loss = 100.0
    train_loss = []
    eval_loss = []
    train_acc = []
    eval_acc = []

    print('start train ..')
    for epoch_id in range(EPOCH_NUM):
        model.train()
        los_list = []
        acc_list = []
        for batch_id, data in enumerate(train_loader()):
            image, label, mmse = data

            predict, out = model(image, mmse)
            loss = loss_fn(predict, label)
            if similarity_loss is not None:
                similar_loss = similarity_loss(out, label)
                loss = loss + loss_rate * similar_loss

            acc = torch.metric.accuracy(predict, label)
            acc_list.extend(acc.numpy())
            los_list.append(loss.numpy())

            loss_str = "epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy())
            print(loss_str)
            logger.write(loss_str + '\n')

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id >= t_rate * len(train_loader):
                break

        epoch_loss = np.mean(los_list)
        epoch_acc = np.mean(acc_list)
        epoch_loss_str = "epoch: {}, acc is {} ,loss is: {}".format(epoch_id, epoch_acc, epoch_loss)
        print(epoch_loss_str)
        logger.write(epoch_loss_str + '\n')
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        model.eval()
        acc_list = []
        los_list = []
        print('testing ...')
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader()):
                x_data, y_data, mmse = data
                predicts, _ = model(x_data, mmse)
                loss = loss_fn(predicts, y_data)
                acc = torch.metric.accuracy(predicts, y_data)
                acc_list.extend(acc.numpy())
                los_list.append(loss.numpy())
                eval_loss_str = "batch_id: {}, acc is: {}".format(batch_id, acc.numpy())
                print(eval_loss_str)
                logger.write(eval_loss_str + '\n')

        val_acc = np.mean(acc_list)
        val_los = np.mean(los_list)
        epoch_eval_loss_str = 'val acc is {} , val loss is {}'.format(val_acc, val_los)
        print(epoch_eval_loss_str)
        logger.write(epoch_eval_loss_str + '\n')
        eval_loss.append(val_los)
        eval_acc.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_model_path)
        elif val_acc == best_acc and val_los < min_loss:
            min_loss = val_los
            torch.save(model.state_dict(), save_model_path)
        if val_acc >= best_acc_actual:
            best_acc_actual = val_acc
        scheduler.step()
        # 早停
        if early_stop is not None and early_stop(val_acc):
            break

    logger.write('best acc is {} \n'.format(best_acc_actual))
    if not logger.closed:
        logger.close()

    return train_loss, eval_loss, train_acc, eval_acc


def test_mmse(model, data_loader, loss_fn):
    preds = []
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        for batch_id, data in enumerate(data_loader()):
            x_data,_,mmse = data
            pred = model(x_data,mmse)
            pred = softmax(pred)
            preds.append(pred.numpy())
    return preds