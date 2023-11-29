from easydict import EasyDict

opt = EasyDict()

# ---- Train configs ----
opt.device = 'cuda:7'
opt.mudule = 'mwp1'
opt.model_name = 'model20231129_122432_only_param_weights'
opt.rslt_path = 'output/'
opt.epoch = 50
opt.endure_epochs = 10
opt.batch_size = 4
opt.last_best_acc = 0.9474
opt.loss_rate = 0.1
opt.patience = 5  # 3-15
opt.dataset = 'data/1'
opt.lr = 1e-5  # 1e-4 is good, it should be set to a lower value in refine stages

# ---- Test configs ----
opt.test = {}
opt.test.print_metric = True
opt.test.dataset = 'data/1'
opt.test.model_name = 'model20231129_122432_only_param_weights'  # may be changed by command line [refer to readme.md]
