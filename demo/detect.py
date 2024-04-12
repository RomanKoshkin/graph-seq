import matplotlib


import sys, os, argparse
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append('../')
from utils.constants import *
import numpy as np
import torch
import torch.optim as optim

from models.models import WeightedGCN
from models.Kmeans import Kmeans
from utils.GNNTrainer import Trainer


from utils.utils import load_config
from utils.utils import Timer



from models.models import WeightedGCN
from models.Kmeans import Kmeans

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import resource
torch.multiprocessing.set_sharing_strategy('file_system') 
print(torch.multiprocessing.get_sharing_strategy())
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


matplotlib.rcParams.update(load_config('../configs/rcparams.yaml'))


parser = argparse.ArgumentParser(description="GNN for spike sequence detection")
parser.add_argument("--snapshot_interval", type=int, help='Interval (epochs) for saving pictures', default=5)
parser.add_argument("--z_dim", type=int, default=None, help='hidden dimensions', required=True)
parser.add_argument("--K", type=int, default=None, help='number of sequeces assumed', required=True)
parser.add_argument("--epochs", type=int, default=None, help='number of epochs to train the model', required=True)
args = parser.parse_args()

params = load_config('../configs/config.yaml')


# load dataset
dataset = np.load('../datasets/DATA.npy', allow_pickle=True).item()
DATA = dataset['DATA']
DATAnull = dataset['DATAnull']
step_sz = dataset['step_sz']
X = dataset['X']
X_ = dataset['X_']

# update parameters
params.snapshot_interval = args.snapshot_interval
params.fs = 1/params.dt # sampling rate
params.Ts = X.shape[1]
params.z_dim = args.z_dim
params.K = args.K

# move data to device
for i in trange(len(DATA), bar_format=bar_format):
    DATA[i] = DATA[i].to(device=params.device)
for i in trange(len(DATAnull), bar_format=bar_format):
    DATAnull[i] = DATAnull[i].to(device=params.device)

# create model, optimizer, scheduler and trainer objects
model = WeightedGCN(params).to(params.device)
kmeans = Kmeans(params, debug=False)
writer = SummaryWriter(log_dir='../runs/')
optimizer = optim.AdamW(model.parameters(), lr=0.05) # tell the optimizer which var we want optimized
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, total_iters=100)
trainer = Trainer(DATA, X_, model, optimizer, scheduler, kmeans, params, step_sz)

fnames = [i for i in os.listdir("../data/") if i.startswith('proj')]
for fn in fnames:
    os.remove(f"../data/{fn}")


trainer.MAX_EPOCHS = args.epochs
with Timer():
    trainer.train()
