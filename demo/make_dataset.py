import matplotlib
import matplotlib.pyplot as plt

import sys, argparse
sys.path.append('../')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from utils.constants import *
import numpy as np
from tqdm import trange


import torch
import scipy.io as sio

from utils.utils import load_config
from utils.graph_conversion import convertToGraphs
from utils.synthetic import get_background2, embed_three_seq

import resource
torch.multiprocessing.set_sharing_strategy('file_system') 
print(torch.multiprocessing.get_sharing_strategy())
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


matplotlib.rcParams.update(load_config('../configs/rcparams.yaml'))

parser = argparse.ArgumentParser(description="GNN for spike sequence detection")
parser.add_argument("--p_drop", type=float, default=None, help='probability of dropping a spike', required=True)
parser.add_argument("--gap_ts", type=int, default=None, help='gap between sequences, time steps', required=True)
parser.add_argument("--seqlen", type=int, default=None, help='number of neurons in a spike sequence', required=True)
parser.add_argument("--jitter_std", type=int, default=None, help='spike timing jitter standard deviation', required=True)
parser.add_argument("--winsize", type=int, default=None, help='window size for graph conversion', required=True)
parser.add_argument("--step_sz", type=int, default=None, help='step size for sliding window for graph conversion', required=True)
parser.add_argument("--tau", type=int, default=None, help='sdecay time constant of spike importance', required=True)
args = parser.parse_args()


# load the parameters from YAML file
params = load_config('../configs/config.yaml')
params.fs = 1/params.dt # sampling rate


# load CA1 data
dat1 = sio.loadmat(f'{params.path_to_data}/position_per_frame.mat')['position_per_frame'].flatten().astype('float')
dat1 = dat1/dat1.max()*96 # make sure the position ranges from 0 to 96 cm (as in the paper)
X = sio.loadmat(f'{params.path_to_data}/neuronal_activity_mat.mat')['neuronal_activity_mat']

params.Ts = X.shape[1]


# sequence properties
p_drop = args.p_drop
gap_ts = args.gap_ts
seqlen = args.seqlen
jitter_std = args.jitter_std

# graph conversion parameters
winsize = args.winsize
step_sz = args.step_sz
tau = args.tau

X_ = get_background2(X, pruning_factor=0.0)
X_, GT = embed_three_seq(X_, params, p_drop, gap_ts, seqlen, jitter_std)
Xnull = get_background2(X_, pruning_factor=0.0)


colors = ['red', 'green', 'cyan']
fig, ax = plt.subplots(3, 1, figsize=(18,7), sharex=True)
ax[0].spy(X, aspect='auto', markersize=1.5, origin='lower')
ax[1].spy(X_, aspect='auto', markersize=1.5, origin='lower')
ax[2].spy(Xnull, aspect='auto', markersize=1.5, origin='lower')

for s, c in zip(GT, colors):
    for i in s:
        ax[1].axvline(i, color=c)
        
ax[0].set_xlim(0, 6000)
plt.savefig('../artifacts/dataset.png', dpi=300)
plt.close('all')

# convert to graphs    
DATA = convertToGraphs(X_, winsize, step_sz, tau, num_workers=20)   
DATAnull = convertToGraphs(Xnull, winsize, step_sz, tau, num_workers=20)


time_gap = np.linspace(0.001, winsize, 100)
halflife = winsize/6
tau = halflife/np.log(2) # samples
exp_wgt = np.exp(-time_gap/tau)

plt.plot(time_gap, exp_wgt)
plt.title(f'tau={tau:.2f}, halflife: {halflife:.3f}')
plt.savefig('../artifacts/spike_importance_decay.png', dpi=300)
plt.close('all')

np.save('../datasets/DATA.npy', 
        {'DATA': DATA, 
         'DATAnull': DATAnull,
         'X': X,
         'X_': X_,
         'step_sz': step_sz})

