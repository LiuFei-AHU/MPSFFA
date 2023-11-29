"""
 This is the Dataset codes for preparing data input into the model in the training or testing stage.
 Created by liufei@liu.jason0728@gmail.com.
"""

from torch.utils.data import Dataset
import torch
import pickle
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, root, data=None, mode="train", input_3d=True, s_norm=False, mean=0.0, std=1.0, device='cpu',
                 aug=None):
        super().__init__()
        if data is None:
            with open(root, 'rb') as f:
                data_dict = pickle.load(f)
            self.images = data_dict[mode][0]
            self.labels = data_dict[mode][1]
        else:
            self.images = data[0]
            self.labels = data[1]
        self.mean = mean
        self.std = std
        self.input_3d = input_3d
        self.s_norm = s_norm
        self.device = device
        self.aug = aug
        self.mode = mode
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        if self.s_norm:
            mn = np.min(img)
            mx = np.max(img)
            img = (img - mn) / (mx - mn)
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, dtype=torch.float32)
        lbl = torch.tensor(lbl, dtype=torch.long)
        # img = img.to(self.device)
        # lbl = lbl.to(self.device)
        if self.input_3d:
            img = img.unsqueeze(dim=0)  # dhw->cdhw
        return img, lbl

    def __len__(self):
        return len(self.labels)


class NDataSet(Dataset):
    def __init__(self, mris, labels, stas, sids, aug=None):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.stas = stas
        self.sids = sids
        self.aug = aug
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.mris[index]
        lbl = self.labels[index]
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        lbl = torch.tensor(lbl, torch.long)
        return img, lbl

    def __len__(self):
        return len(self.labels)


class NDataSet_Mmse(Dataset):
    def __init__(self, mris, labels, stas, sids, aug=None):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.stas = stas
        self.sids = sids
        self.aug = aug
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.mris[index]
        lbl = self.labels[index]
        mmse = float(self.stas[index][6]) / 30.0
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        lbl = torch.tensor(lbl, dtype=torch.long)
        mmse = torch.tensor(mmse, dtype=torch.float32)
        return img, lbl, mmse

    def __len__(self):
        return len(self.labels)


class NDataSet_Mmse_Age(Dataset):
    def __init__(self, mris, labels, stas, sids, aug=None):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.stas = stas
        self.sids = sids
        self.aug = aug
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.mris[index]
        lbl = self.labels[index]
        mmse = float(self.stas[index][6]) / 30.0
        age = float(self.stas[index][2]) / 100.0
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        lbl = torch.tensor(lbl, torch.long)
        mmse = torch.tensor(mmse, torch.float32)
        age = torch.tensor(age, torch.float32)
        return img, lbl, mmse, age

    def __len__(self):
        return len(self.labels)


class NDataSet_CDRsb(Dataset):
    def __init__(self, mris, labels, stas, sids, aug=None):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.stas = stas
        self.sids = sids
        self.aug = aug
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.mris[index]
        lbl = self.labels[index]
        cdr = float(self.stas[index][7]) / 18.0
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        lbl = torch.tensor(lbl, torch.long)
        cdr = torch.tensor(cdr, torch.float32)
        return img, lbl, cdr

    def __len__(self):
        return len(self.labels)


class NDataSet_CDRsb_Age(Dataset):
    def __init__(self, mris, labels, stas, sids, aug=None):
        super().__init__()
        self.mris = mris
        self.labels = labels
        self.stas = stas
        self.sids = sids
        self.aug = aug
        assert aug is None or isinstance(self.aug, (list, tuple, set)), 'aug object must be list of DataAug!'

    def __getitem__(self, index):
        img = self.mris[index]
        lbl = self.labels[index]
        cdr = float(self.stas[index][7]) / 18.0
        age = float(self.stas[index][2]) / 100.0
        if self.aug:
            for aug in self.aug:
                img = aug(img)
        img = torch.tensor(img, torch.float32).unsqueeze(dim=0)  # dhw->cdhw
        lbl = torch.tensor(lbl, torch.long)
        cdr = torch.tensor(cdr, torch.float32)
        age = torch.tensor(age, torch.float32)
        return img, lbl, cdr, age

    def __len__(self):
        return len(self.labels)

