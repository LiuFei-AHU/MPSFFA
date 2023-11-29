import pickle
import numpy as np
import math
import random
import os


def load_adni(r_path, cats=('AD', 'CN'), lbl=(1, 0)):
    sids = []
    mris = []
    stas = []
    labels = []
    for g in ['ADNI1', 'ADNI2', 'ADNI3']:
        for c in cats:
            f_path = f'{r_path}/{g}-{c}.npy'
            if not os.path.exists(f_path):
                continue  # skip when file is not exists
            with open(f_path, 'rb') as f:
                data = pickle.load(f)  # (mri,[])
                sids.extend(list(data.keys()))
                data = list(data.values())
                for mri, sta in data:
                    mris.append(mri)
                    stas.append(sta)
                label = [lbl[cats.index(c)] for _ in data]
                labels.extend(label)
                print('{}-{} contains {} mris'.format(g, c, len(data)))

    print('Dataset length: ', len(sids), len(mris), len(stas), len(labels))

    # fix the random seed
    np.random.seed(2022)
    index = np.random.permutation(np.arange(len(mris)))
    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    mris, labels, stas, sids = mris[index], labels[index], stas[index], sids[index]
    return mris, labels, stas, sids


def load_aibl(r_path, cats=('AD', 'CN'), lbl=(1, 0)):
    sids = []
    mris = []
    stas = []
    labels = []
    g = 'AIBL'
    for c in cats:
        f_path = f'{r_path}/{g}-{c}.npy'
        with open(f_path, 'rb') as f:
            data = pickle.load(f)  # (mri,[])
            sids.extend(list(data.keys()))
            data = list(data.values())
            for mri, sta in data:
                mris.append(mri)
                stas.append(sta)
            label = [lbl[cats.index(c)] for _ in data]
            labels.extend(label)
            print('{}-{} contains {} mris'.format(g, c, len(data)))

    print('Dataset length: ', len(sids), len(mris), len(stas), len(labels))

    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    return mris, labels, stas, sids


def load_adni_balance(r_path, cats=('AD', 'CN'), lbl=(1, 0), rate=1.1):
    """
    Sample the data by groups.
    :param r_path: base path of the data
    :param cats: groups you want to pick
    :param lbl: group labels
    :param rate: sample rate
    :return: mris, labels, stas, sids
    """
    sids_d = {}
    mris_d = {}
    stas_d = {}
    labels_d = {}
    for g in ['ADNI1', 'ADNI2', 'ADNI3']:
        for c in cats:
            f_path = f'{r_path}/{g}-{c}.npy'
            if not os.path.exists(f_path):
                continue
            if c not in sids_d.keys():
                sids_d[c] = []
                mris_d[c] = []
                stas_d[c] = []
                labels_d[c] = []
            with open(f_path, 'rb') as f:
                data = pickle.load(f)  # (mri,[])
                sids_d[c].extend(list(data.keys()))
                data = list(data.values())
                for mri, sta in data:
                    mris_d[c].append(mri)
                    stas_d[c].append(sta)
                label = [lbl[cats.index(c)] for _ in data]
                labels_d[c].extend(label)
                print('{}-{} contains {} mris'.format(g, c, len(data)))

    # fix the random seed
    np.random.seed(2022)
    random.seed(2022)

    cat_nums = [len(sids_d[c]) for c in cats]
    mn = min(cat_nums)
    sm_len = math.floor(mn * rate)

    for c in cats:
        if len(sids_d[c]) / mn > rate:
            sids_d[c] = random.sample(sids_d[c], sm_len)
            mris_d[c] = random.sample(mris_d[c], sm_len)
            stas_d[c] = random.sample(stas_d[c], sm_len)
            labels_d[c] = random.sample(labels_d[c], sm_len)

    mris, labels, stas, sids = [], [], [], []
    for c in cats:
        mris.extend(mris_d[c])
        labels.extend(labels_d[c])
        stas.extend(stas_d[c])
        sids.extend(sids_d[c])

    print('Dataset length: ', len(sids), len(mris), len(stas), len(labels))

    # shuffle the dataset
    index = np.random.permutation(np.arange(len(mris)))
    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    mris, labels, stas, sids = mris[index], labels[index], stas[index], sids[index]
    return mris, labels, stas, sids


def load_adni_balance2(r_path, cats=('AD', 'CN'), lbl=(1, 0), rate=1.1):
    sids_d = {}
    mris_d = {}
    stas_d = {}
    labels_d = {}
    for g in ['ADNI1', 'ADNI2', 'ADNI3']:
        for c in cats:
            f_path = f'{r_path}/{g}-{c}.npy'
            if not os.path.exists(f_path):
                continue
            if c not in sids_d.keys():
                sids_d[c] = []
                mris_d[c] = []
                stas_d[c] = []
                labels_d[c] = []
            with open(f_path, 'rb') as f:
                data = pickle.load(f)  # (mri,[])
                sids_d[c].extend(list(data.keys()))
                data = list(data.values())
                for mri, sta in data:
                    mris_d[c].append(mri)
                    stas_d[c].append(sta)
                label = [lbl[cats.index(c)] for _ in data]
                labels_d[c].extend(label)
                print('{}-{} contains {} mris'.format(g, c, len(data)))

    # fix the random seed
    np.random.seed(2022)
    random.seed(2022)

    cat_nums = [len(sids_d[c]) for c in cats]
    mn = min(cat_nums)
    sm_len = math.floor(mn * rate)

    for c in cats:
        if len(sids_d[c]) / mn > rate:
            idx = np.random.randint(0, len(sids_d[c]), sm_len)
            mris_d[c], labels_d[c] = [mris_d[c][i] for i in idx], [labels_d[c][i] for i in idx]
            stas_d[c], sids_d[c] = [stas_d[c][i] for i in idx], [sids_d[c][i] for i in idx]

    mris, labels, stas, sids = [], [], [], []
    for c in cats:
        mris.extend(mris_d[c])
        labels.extend(labels_d[c])
        stas.extend(stas_d[c])
        sids.extend(sids_d[c])

    print('Dataset length: ', len(sids), len(mris), len(stas), len(labels))

    # shuffle the dataset
    index = np.random.permutation(np.arange(len(mris)))
    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    mris, labels, stas, sids = mris[index], labels[index], stas[index], sids[index]
    return mris, labels, stas, sids


def load_adni_pmci_balance(r_path, pmci_ids, smci_ids, cats=('PMCI', 'SMCI'), lbl=(1, 0), rate=1.1):
    sids_d = {}
    mris_d = {}
    stas_d = {}
    labels_d = {}
    for g in ['ADNI1', 'ADNI2', 'ADNI3']:
        for c in cats:
            f_path = f'{r_path}/{g}-{c}.npy'
            if not os.path.exists(f_path):
                continue
            if c not in sids_d.keys():
                sids_d[c] = []
                mris_d[c] = []
                stas_d[c] = []
                labels_d[c] = []
            with open(f_path, 'rb') as f:
                data = pickle.load(f)  # (mri,[])
                if c == 'PMCI':
                    data = {key: data[key] for key in data if key in pmci_ids}
                else:
                    data = {key: data[key] for key in data if key in smci_ids}
            sids_d[c].extend(list(data.keys()))
            data = list(data.values())
            for mri, sta in data:
                mris_d[c].append(mri)
                stas_d[c].append(sta)
            label = [lbl[cats.index(c)] for _ in data]
            labels_d[c].extend(label)
            print('{}-{} contains {} mris'.format(g, c, len(data)))

    # fix the random seed
    np.random.seed(2022)
    random.seed(2022)

    cat_nums = [len(sids_d[c]) for c in cats]
    mn = min(cat_nums)
    sm_len = math.floor(mn * rate)

    for c in cats:
        if len(sids_d[c]) / mn > rate:
            sids_d[c] = random.sample(sids_d[c], sm_len)
            mris_d[c] = random.sample(mris_d[c], sm_len)
            stas_d[c] = random.sample(stas_d[c], sm_len)
            labels_d[c] = random.sample(labels_d[c], sm_len)

    mris, labels, stas, sids = [], [], [], []
    for c in cats:
        mris.extend(mris_d[c])
        labels.extend(labels_d[c])
        stas.extend(stas_d[c])
        sids.extend(sids_d[c])

    print('Dataset length: ', len(sids), len(mris), len(stas), len(labels))

    # shuffle the dataset
    index = np.random.permutation(np.arange(len(mris)))
    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    mris, labels, stas, sids = mris[index], labels[index], stas[index], sids[index]
    return mris, labels, stas, sids


def load_adni_pmci_balance2(pmci_ids, smci_ids, cats=('PMCI', 'SMCI'), lbl=(1, 0)):
    sids_d = {}
    mris_d = {}
    stas_d = {}
    labels_d = {}
    for g in ['ADNI1', 'ADNI2', 'ADNI3']:
        for c in cats:
            # 数据路径
            f_path = r'/home/aistudio/data/data143846/{}-{}.npy'.format(g, c)
            if not os.path.exists(f_path):
                continue
            # 初始类别列表
            if not c in sids_d.keys():
                sids_d[c] = []
                mris_d[c] = []
                stas_d[c] = []
                labels_d[c] = []
            # 读写数据
            with open(f_path, 'rb') as f:
                data = pickle.load(f)  # (mri,[])
                # 按要求删除
                if c == 'PMCI':
                    data = {key: data[key] for key in data if key in pmci_ids}
                else:
                    data = {key: data[key] for key in data if key in smci_ids}
            sids_d[c].extend(list(data.keys()))
            data = list(data.values())
            for mri, sta in data:
                mris_d[c].append(mri)
                stas_d[c].append(sta)
            label = [lbl[cats.index(c)] for _ in data]
            labels_d[c].extend(label)
            print('{}-{} contains {} mris'.format(g, c, len(data)))

    # 固定种子
    np.random.seed(2022)
    random.seed(2022)

    # 对数据多的类别采样，采样率为少的类别的1.1倍
    cat_nums = [len(sids_d[c]) for c in cats]
    mn = min(cat_nums)
    sm_len = math.floor(mn * 1.1)

    for c in cats:
        if len(sids_d[c]) / mn > 1.1:
            idx = np.random.randint(0, len(sids_d[c]), sm_len)
            mris_d[c], labels_d[c] = [mris_d[c][i] for i in idx], [labels_d[c][i] for i in idx]
            stas_d[c], sids_d[c] = [stas_d[c][i] for i in idx], [sids_d[c][i] for i in idx]

    mris, labels, stas, sids = [], [], [], []
    for c in cats:
        mris.extend(mris_d[c])
        labels.extend(labels_d[c])
        stas.extend(stas_d[c])
        sids.extend(sids_d[c])

    print('数据集长度：', len(sids), len(mris), len(stas), len(labels))

    # 随机打乱
    index = np.random.permutation(np.arange(len(mris)))
    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    mris, labels, stas, sids = mris[index], labels[index], stas[index], sids[index]
    return mris, labels, stas, sids


# aibl:按要求过滤数据
def load_aibl_pmci_balance(pmci_ids, smci_ids, cats=['PMCI', 'SMCI'], lbl=[1, 0]):
    sids = []
    mris = []
    stas = []
    labels = []
    g = 'AIBL'
    for c in cats:
        f_path = r'/home/aistudio/data/data143846/{}-{}.npy'.format(g, c)
        if not os.path.exists(f_path):
            continue
        # 读写数据
        with open(f_path, 'rb') as f:
            data = pickle.load(f)  # (mri,[])
            # 按要求删除
            if c == 'PMCI':
                if pmci_ids is not None and len(pmci_ids) > 0:
                    data = {key: data[key] for key in data if key in pmci_ids}
            else:
                if smci_ids is not None and len(smci_ids) > 0:
                    data = {key: data[key] for key in data if key in smci_ids}
        sids.extend(list(data.keys()))
        data = list(data.values())
        for mri, sta in data:
            mris.append(mri)
            stas.append(sta)
        label = [lbl[cats.index(c)] for _ in data]
        labels.extend(label)
        print('{}-{} contains {} mris'.format(g, c, len(data)))

    print('数据集长度：', len(sids), len(mris), len(stas), len(labels))

    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    return mris, labels, stas, sids


# aibl:取不在列表中的文件
def load_aibl_smci_reverse(smci_ids, cats=['SMCI'], lbl=[0]):
    sids = []
    mris = []
    stas = []
    labels = []
    g = 'AIBL'
    for c in cats:
        f_path = r'/home/aistudio/data/data143846/{}-{}.npy'.format(g, c)
        if not os.path.exists(f_path):
            continue
        # 读写数据
        with open(f_path, 'rb') as f:
            data = pickle.load(f)  # (mri,[])
            # 按要求删除
            if c == 'SMCI':
                if smci_ids is not None and len(smci_ids) > 0:
                    data = {key: data[key] for key in data if key not in smci_ids}
        sids.extend(list(data.keys()))
        data = list(data.values())
        for mri, sta in data:
            mris.append(mri)
            stas.append(sta)
        label = [lbl[cats.index(c)] for _ in data]
        labels.extend(label)
        print('{}-{} contains {} mris'.format(g, c, len(data)))

    print('数据集长度：', len(sids), len(mris), len(stas), len(labels))

    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    return mris, labels, stas, sids


# aibl:读取手工提取的pmci/smci数据
def load_aibl_pmci2(cats=['PMCI', 'SMCI'], lbl=[1, 0]):
    sids = []
    mris = []
    stas = []
    labels = []
    g = 'AIBL'
    for c in cats:
        f_path = r'/home/aistudio/work/data/{}-{}2.npy'.format(g, c)
        if not os.path.exists(f_path):
            continue
        with open(f_path, 'rb') as f:
            data = pickle.load(f)  # (mri,[])
            sids.extend(list(data.keys()))
            data = list(data.values())
            for mri, sta in data:
                mris.append(mri)
                stas.append(sta)
            label = [lbl[cats.index(c)] for _ in data]
            labels.extend(label)
            print('{}-{} contains {} mris'.format(g, c, len(data)))

    print('数据集长度：', len(sids), len(mris), len(stas), len(labels))

    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    return mris, labels, stas, sids


# 正例采样
def load_aibl_sample(sample_rate=0.2, cats=['AD', 'CN'], lbl=[1, 0]):
    sids = []
    mris = []
    stas = []
    labels = []
    g = 'AIBL'
    for c in cats:
        f_path = r'/home/aistudio/data/data143754/{}-{}.npy'.format(g, c)
        with open(f_path, 'rb') as f:
            data = pickle.load(f)  # (mri,[])
            sids.extend(list(data.keys()))
            data = list(data.values())
            for mri, sta in data:
                mris.append(mri)
                stas.append(sta)
            label = [lbl[cats.index(c)] for _ in data]
            labels.extend(label)
            print('{}-{} contains {} mris'.format(g, c, len(data)))

    idx = np.random.randint(0, len(data), math.floor(len(data) * sample_rate))
    mris, labels, stas, sids = [mris[i] for i in idx], [labels[i] for i in idx], [stas[i] for i in idx], [sids[i] for i
                                                                                                          in idx]
    print('数据集长度：', len(sids), len(mris), len(stas), len(labels))

    mris, labels, stas, sids = np.asarray(mris), np.asarray(labels), np.asarray(stas), np.asarray(sids)
    return mris, labels, stas, sids
