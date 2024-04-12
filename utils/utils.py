from itertools import chain, groupby
from collections import Counter
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

import sys, time, json, yaml, subprocess, copy
from tqdm import trange
from utils.constants import bar_format

import os, pickle, shutil, warnings, shutil
import numpy as np
from multiprocessing import Pool
from scipy import signal
import numba
from numba import njit, jit, prange
from numba.typed import List

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from termcolor import cprint
from pprint import pprint

import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph
from collections import namedtuple

from utils.metrics import get_metrics

from torch_geometric.loader import DataLoader
from models.models import WeightedGCN

import sknetwork
if sknetwork.__version__ == '0.20.0':
    SKNETWORK_OLD = True
    from sknetwork.clustering import Louvain, modularity
elif sknetwork.__version__ == '0.30.0':
    from sknetwork.clustering import Louvain
    from sknetwork.clustering import get_modularity as modularity
    SKNETWORK_OLD = False
else:
    raise ValueError("You must have scikit-network of the following versions: 0.20.0 or 0.30.0")


class Timer:

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time
        print(f'{self.interval:.4f} s.')


class HiddenPrints:

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class AttributeDict(dict):
    """ convenience class. To get and set properties both as if it were a dict or obj """
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def load_config(pathAndNameOfConfig):
    yaml_txt = open(pathAndNameOfConfig).read()
    parms_dict = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    parms_obj = AttributeDict()
    for k, v in parms_dict.items():
        parms_obj[k] = v
    return parms_obj


def getOutputEmbeddingSimMatrix(DATA, winsize, model=None, quantile=0.99, params=None):
    if model is None:
        print('computing from a random model')
        with HiddenPrints():
            model = WeightedGCN(params).to(params.device)  # build a null model
    loader = DataLoader(DATA, batch_size=len(DATA), shuffle=False)
    for batch in loader:
        pass
    ems, logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch, deterministic=False)
    sm_torch = torch.matmul(ems, ems.T)
    triu_idx = torch.triu_indices(sm_torch.shape[0], sm_torch.shape[1], offset=winsize * 2)
    smnp = tonp(sm_torch.squeeze())
    triu = smnp[triu_idx[0], triu_idx[1]].flatten()
    q = np.quantile(triu, quantile)
    return smnp, q, triu


def diff(x, dim=0):
    if dim == 1:
        return x[:, 1:] - x[:, :-1]
    else:
        raise NotImplementedError('Not implemented')


def highlight_above_q99(x, q_99, ax, xlim, highlight=True):
    A = []
    started = False

    for i in range(x.shape[0]):

        if not started:
            if x[i] > q_99:
                st = i
                started = True
        if started:
            if x[i] < q_99:
                en = i
                A.append((st, en - 1))
                started = False

    for it in A:
        if it[1] - it[0] >= 10:
            mid = np.round(np.mean(it)).astype(int)
            if mid < xlim[1]:
                if highlight:
                    ax.axvspan(*it, color='red', alpha=0.4)
                ax.plot(mid, x[mid] + 5, "k*")

    return A


def tonp(x):
    return x.cpu().detach().numpy()


def get_significants(ids, Z, class_id):
    # get span lengths get continous spans of class labels
    idx0 = np.where((ids == class_id).cpu().numpy())[0]
    idx0_spans = get_continuous_indices(np.diff(idx0))

    # get span lengths
    for i, span in enumerate(idx0_spans):
        idx0_spans[i] += (span[1] - span[0],)

    # sort by width, widest will come first
    idx0_spans = sorted(idx0_spans, key=lambda tup: tup[2], reverse=True)

    span_lengths = np.array([i[2] for i in idx0_spans])
    median = np.median(span_lengths)
    median_len_span_idx = np.argmin(span_lengths - median)

    # span = idx0_spans[median_len_span_idx] # take the span with the median width as a reference
    span = idx0_spans[0]  # take the span with the largets width as a reference
    st, en = span[0], span[1]
    significants = np.where(Z[idx0[st]:idx0[en], :].mean(0) > 0.01)[0]  # at least half of the points
    return significants


def getLimitsOfContigElements(arr, highest_threshold):
    """
    RETURNS:
        start and stop indices of elements that are lower than the threshold
    """
    org_idx = np.where(arr < highest_threshold)[0]  # get indices of items in the orig array that satisfy the condition
    diffs = np.diff(org_idx)  # get diffs
    lims = get_continuous_indices(diffs)  # get indices of consecutive items that satisfy the condition
    return [(org_idx[st], org_idx[en]) for st, en in lims]


def get_continuous_indices(arr):
    """ 
    ARGS:
        np array of int or bools
        e.g. 
            arr = np.array([1,1,2,3,4,5,6,6,6,6,0,2,2]), or 

    RETURNS:
        a list of tuples [(st, en),...,(st, en)] of same subarrays
    """
    segment_indices = []
    for k, g in groupby(enumerate(arr), key=lambda x: x[1]):
        segment = list(g)
        if len(segment) > 1:
            segment_indices.append((segment[0][0], segment[-1][0] + 1))
    return segment_indices


def remap_to_ordered_ids(ids_tensor):
    """ 
    maps cluster ids to temporally ordered ids for intelligible plotting
    """
    idslist = ids_tensor.clone().tolist()
    Maxid = max(np.unique(idslist))

    dic = {}
    j = 0
    for i in idslist:
        if i not in dic.keys():
            dic[i] = j
            j += 1
        else:
            pass
    return list(map(lambda x: abs(dic[x] - Maxid), idslist))


def get_labels(dat1, params, DATA, step_sz):
    behav = dat1 / 100 * params.K
    b = np.zeros_like(behav)
    b[(behav > 0.5) & (behav < 2.7)] = 1

    switched = False
    up = True

    for i in range(len(b)):
        if b[i] != 0:
            switched = False
            if up:
                b[i] = 1.0
            else:
                b[i] = -1.0

        if (b[i] == 0):
            if not switched:
                up = not up
                switched = True
    x = np.arange(0, len(DATA) * step_sz, step_sz)
    labels = b[0::step_sz][:len(x)]
    return x, labels


def saveForPPSeq(X_, cellid=0, path='datasets', name='ds'):
    with open(f'{path}/{name}_{cellid}.txt', 'w') as f:
        for i in range(X_.shape[0]):
            for j in range(X_.shape[1]):
                if X_[i, j] == 1:
                    f.write(f"{float(i+1)}\t{float(j)}\n")


def clusterize(w):
    x = np.copy(w)
    G = nx.from_numpy_matrix(x)
    adj_mat = nx.to_numpy_array(G)
    louvain = Louvain()
    if SKNETWORK_OLD:
        labels = louvain.fit_transform(adj_mat)
    else:
        labels = louvain.fit_predict(adj_mat)
    mod = modularity(adj_mat, labels)

    labels_unique, counts = np.unique(labels, return_counts=True)

    tmp = sorted([(i, d) for i, d in enumerate(labels)], key=lambda tup: tup[1], reverse=True)
    newids = [i[0] for i in tmp]

    W_ = x
    W_ = W_[newids, :]
    W_ = W_[:, newids]
    return W_, labels, counts, mod, newids