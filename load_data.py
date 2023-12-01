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
