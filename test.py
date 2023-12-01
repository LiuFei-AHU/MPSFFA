import os
import numpy as np
import torch
from model import MPSFFA
from dataset import NDataSet_Mmse as MyDataset
from train import test_mmse as test
from load_data import load_aibl
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from eval import metric
import config
import warnings
warnings.filterwarnings("ignore")

# load configs
opt = config.opt
# set device
if torch.cuda.is_available():
    device = torch.device(opt.device)
    torch.cuda.set_device(device)
    print('Running on the GPU')
else:
    device = torch.device('cpu')
    # torch.cuda.set_device(torch.device('cpu'))
    print('Running on the CPU')

opt.device = device

# create model and load params
model = MPSFFA()
model.to(device)
params_file_path = os.path.join(opt.rslt_path, opt.mudule, opt.test.model_name + '.pdparams')
if os.path.exists(params_file_path):
    model.load_state_dict(torch.load(params_file_path, map_location=opt.device))
    print('model loaded!')

# load test dataset
mris, labels, stas, sids = load_aibl(r_path=opt.test.dataset)
ds = MyDataset(mris, labels, stas, sids)
data_loader = DataLoader(ds, shuffle=False, batch_size=opt.batch_size)

# run test code
pred = test(opt, model, data_loader)

if opt.test.print_metric:
    pred = np.asarray(pred)
    y_true = labels
    pred_label = np.argmax(pred, axis=-1).reshape(pred.shape[0], )
    # pred_max = np.max(pred, axis=-1)
    # print(pred_label)
    # print(pred_max)

    acc, f1, auc = accuracy_score(y_true, pred_label), f1_score(y_true, pred_label), roc_auc_score(y_true, pred_label)
    print(acc, f1, auc)
    acc, spe, sen = metric(y_true, pred_label)
    print(acc, spe, sen)
