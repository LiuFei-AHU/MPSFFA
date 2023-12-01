"""
 This is a collection of codes used in the training or testing stage.
 Created by liufei@liu.jason0728@gmail.com.
"""
import torch
import numpy as np
from utils import LogisticLoss
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm


def train(model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop, save_model_path,
          last_best_acc=0.5):
    loss_fn = torch.nn.CrossEntropyLoss()
    metric = BinaryAccuracy()
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
            image, label = data

            predict = model(image)
            loss = loss_fn(predict, label)

            acc = metric.accuracy(predict, label)
            acc_list.extend(acc.numpy())
            los_list.append(loss.numpy())

            loss_str = "epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy())
            print(loss_str)
            logger.write(loss_str + '\n')

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

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
                x_data = data[0]
                y_data = data[1]
                predicts = model(x_data)
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


def train_mmse(model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop, save_model_path,
               last_best_acc=0.5, weight=None, t_rate=1):
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    metric = BinaryAccuracy()
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

            acc = metric.accuracy(predict, label)
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
                acc = metric.accuracy(predicts, y_data)
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


def train_mmse_similarity(opt, model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop,
                          save_model_path, last_best_acc=0.5, similarity_loss=None, loss_rate=0.01, weight=None,
                          t_rate=1):
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    metric = BinaryAccuracy()
    metric.to(opt.device)
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
        for batch_id, data in enumerate(train_loader):
            image, label, mmse = data
            image, label, mmse = image.to(opt.device), label.to(opt.device), mmse.to(opt.device)

            predict, out = model(image, mmse)
            loss = loss_fn(predict, label)
            if similarity_loss is not None:
                similar_loss = similarity_loss(out, label)
                loss = loss + loss_rate * similar_loss

            acc = metric(torch.argmax(predict, 1), label)
            loss_val = loss.detach().cpu().numpy()
            acc_list.append(acc.detach().cpu())
            los_list.append(loss_val)

            loss_str = "epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss_val)
            print(loss_str)
            logger.write(loss_str + '\n')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
            for batch_id, data in enumerate(test_loader):
                x_data, y_data, mmse = data
                x_data, y_data, mmse = x_data.to(opt.device), y_data.to(opt.device), mmse.to(opt.device)
                predicts, _ = model(x_data, mmse)
                loss = loss_fn(predicts, y_data)
                acc = metric(torch.argmax(predicts, 1), y_data)
                acc_list.append(acc.detach().cpu())
                los_list.append(loss.detach().cpu().numpy())
                eval_loss_str = "batch_id: {}, acc is: {}".format(batch_id, acc)
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


def train_mmse_l(model, train_loader, test_loader, scheduler, optimizer, logger, EPOCH_NUM, early_stop, save_model_path,
                 last_best_acc=0.5, weight=None):
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    loss_lg = LogisticLoss()
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

            predict, mmse_bl = model(image, mmse, True)
            loss = loss_fn(predict, label) + 0.1 * loss_lg(mmse, mmse_bl)

            acc = torch.metric.accuracy(predict, label)
            acc_list.extend(acc.numpy())
            los_list.append(loss.numpy())

            loss_str = "epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, loss.numpy())
            print(loss_str)
            logger.write(loss_str + '\n')

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

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
                predicts, mmse_bl = model(x_data, mmse, True)
                loss = loss_fn(predicts, y_data) + 0.1 * loss_lg(mmse, mmse_bl)
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


def test(model, data_loader):
    preds = []
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        for batch_id, data in enumerate(data_loader()):
            x_data = data[0]
            pred = model(x_data)
            pred = softmax(pred)
            preds.append(pred.numpy())
    return preds


def test_mmse(opt, model, data_loader):
    preds = []
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(data_loader)):
            x_data, _, mmse = data
            x_data, mmse = x_data.to(opt.device), mmse.to(opt.device)
            pred,_ = model(x_data, mmse)
            pred = softmax(pred)
            preds.extend(pred.detach().cpu().numpy())
    return preds
