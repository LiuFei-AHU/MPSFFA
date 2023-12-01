"""
 Train the MPS-FFA here!
"""
from model import MPSFFA
from dataset import NDataSet_Mmse as MyDataset
from load_data import load_adni_balance
from utils import EarlyStopping, StepDownCosDecay
import torch
import os
import pickle
import time
import math
from train import train_mmse_similarity as train
from torch.utils.data import DataLoader
import config
from utils import CosineSimilarityLoss
import warnings

warnings.filterwarnings("ignore")

# current time stamp string
timestamp = time.strftime('%Y%m%d_%H%M%S')

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
os.makedirs(os.path.join(opt.rslt_path, opt.mudule), exist_ok=True)
params_file_path = os.path.join(opt.rslt_path, opt.mudule, opt.model_name + '.pdparams')
save_model_path = os.path.join(opt.rslt_path, opt.mudule, 'model' + timestamp + '.pdparams')
if os.path.exists(params_file_path):
    model.load_state_dict(torch.load(params_file_path, map_location=opt.device))
    print('model loaded!')

# load dataset
# randomly augment the dataset during training.
# you can input this 'aug' object as a parameter of the customized dataset implementation.
# aug = [xyz_rotate(-5,6,rate=0.2),mask(5,intersect=False)]

# You can split the original dataset into several subsets for training, validating or testing purpose.
# Here is just a demo with only two subsets: train set and test set.
mris, labels, stas, sids = load_adni_balance(r_path=opt.dataset)
l = math.floor(len(mris) * 0.8)

ds = MyDataset(mris[0:l], labels[0:l], stas[0:l], sids[0:l])
train_loader = DataLoader(ds, shuffle=True, batch_size=opt.batch_size)
ds2 = MyDataset(mris[l:], labels[l:], stas[l:], sids[l:])
test_loader = DataLoader(ds2, shuffle=False, batch_size=opt.batch_size)

# setting the optimizer and learning rate
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)
scheduler = StepDownCosDecay(learning_rate=opt.lr, T_max=5, eta_min=1e-6, last_epoch=-1,
                             endure_epochs=opt.endure_epochs, verbose=True)

# this is a simple implementation for recording the outputs when training the model.
# you can change it to other implementations if you like.
log_pth = os.path.join(opt.rslt_path, opt.mudule, 'log', 'train_loss')
os.makedirs(log_pth, exist_ok=True)
logger = open(os.path.join(log_pth, timestamp + '.log'), mode='w', encoding='utf8')

# early stop: When the model cannot learn new knowledge, just stop this process.
# Then may be you can try new ideas to improve it.
early_stop = EarlyStopping(patience=opt.patience)

# similarity_loss as an additional loss function to help the model learn features from different groups.
similarity_loss = CosineSimilarityLoss()
similarity_loss.to(device)

# start training process here!
train_loss, eval_loss, train_acc, eval_acc = train(opt, model, train_loader, test_loader, scheduler, optimizer, logger,
                                                   opt.epoch, early_stop, save_model_path, opt.last_best_acc,
                                                   similarity_loss, opt.loss_rate)

# save results of the training process. if you have saved these data in the processing steps, please overlook it.
run_pth = os.path.join(opt.rslt_path, opt.mudule, 'run_state')
os.makedirs(run_pth, exist_ok=True)
with open(os.path.join(run_pth, 'train_loss.npy'), 'wb') as f:
    pickle.dump(train_loss, f)
with open(os.path.join(run_pth, 'eval_loss.npy'), 'wb') as f:
    pickle.dump(eval_loss, f)
with open(os.path.join(run_pth, 'train_acc.npy'), 'wb') as f:
    pickle.dump(train_acc, f)
with open(os.path.join(run_pth, 'eval_acc.npy'), 'wb') as f:
    pickle.dump(eval_acc, f)
