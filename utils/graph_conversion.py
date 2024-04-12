import torch
import numpy as np
from itertools import repeat
from torch_geometric.data import Data
from torch_geometric.profile import get_data_size
from scipy.sparse import csr_matrix, find
from multiprocessing import Pool
from tqdm import tqdm
from .constants import bar_format
from termcolor import cprint


def convertToGraphs(X, winsize, step_sz, tau, num_workers=20):
    """ 
        Args
            X - dense binary matrix of spikes (N, T)
            winsize - within what window (samples) to consider spike for each datapoint
            step_sz - number of steps (in samples) by which to shift each window
            tau - rate by which to decay the edge weight as a function of time gap between nodes (spikes)
            num_workers - set to a sensible number (less than the number of cores you have)
        Returns
            list of Data objects that Pytorch Geometric understands
    """
    Xsp = csr_matrix(X)
    idx = list(range(0, Xsp.shape[1] - winsize, step_sz))
    sparse_matrix_generator = (Xsp[:, i:i + winsize] for i in idx)

    plist = zip(sparse_matrix_generator, repeat(tau))
    with Pool(20) as p:
        DATA = list(p.starmap(makeGraphDataPointSp, tqdm(plist, bar_format=bar_format, total=len(idx)), chunksize=20))
    dataset_sz_MB = 0
    for d in tqdm(DATA, bar_format=bar_format):
        dataset_sz_MB += get_data_size(d) / 1024**2
    print(f'dataset_sz_MB: {dataset_sz_MB:.2f} | num_datapoints: {len(DATA)}')
    return DATA


def makeGraphDataPoint(xx, tau):
    """ Args
            binary matrix of shape (num_neurons, num_timestep) 
            tau - give exponentially less weight to more distant spikes 
    """

    # get neuronids and spiketimes
    nids, ts = np.where(xx == 1)

    # make edges:
    EDGES = []
    for t in ts:
        n_that_spike_at_t = nids[np.where(ts == t)[0]]
        for n in n_that_spike_at_t:
            idx = np.where((nids != n) & (ts > t))[0]  # all the spike on other neurons that spike later ()!!!!
            time_gap = ts[idx] - t
            exp_wgt = np.exp(-time_gap / tau)
            edges_from_this_neuron = list(
                # nid_from, t_from, nid_to, t_to, t_gap, exp_wgt
                zip(repeat(n), repeat(t), nids[idx], ts[idx], time_gap, exp_wgt))
            EDGES += edges_from_this_neuron

    # make a weighted adjacency matrix
    D = np.zeros(shape=(xx.shape[0], xx.shape[0]))
    for e in EDGES:
        D[e[2], e[0]] += e[5]

    edge_index = []
    edge_weights = []
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if D[i, j] > 0:
                edge_index.append([j, i])  # 2 x T array (top row: from, bottom row: to)
                edge_weights.append(D[i, j])
    edge_index = torch.tensor(np.vstack(edge_index).T, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(np.unique(edge_index), dtype=torch.long).view(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

    mapdict = {it.item(): i for i, it in enumerate(data.x)}

    ei = torch.zeros_like(data.edge_index)
    ei[0, :] = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.edge_index[0, :])))
    ei[1, :] = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.edge_index[1, :])))
    d = torch.zeros_like(data.x)
    d = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.x)))

    tmp = data.edge_index.clone()
    data.edge_index = ei.clone()
    ei = tmp

    return data, D, ei, d, EDGES


def makeGraphDataPointSp(xx, tau):
    """ Args
            binary SPARSE matrix of shape (num_neurons, num_timestep) 
            tau - give exponentially less weight to more distant spikes 
            
    """

    # get neuronids and spiketimes
    #     nids, ts = np.where(xx == 1)
    B = xx == 1
    nids, ts, _ = find(B)

    # make edges:
    EDGES = []
    for t in ts:
        n_that_spike_at_t = nids[np.where(ts == t)[0]]
        for n in n_that_spike_at_t:
            idx = np.where((nids != n) & (ts > t))[0]  # all the spike on other neurons that spike later ()!!!!
            time_gap = ts[idx] - t

            exp_wgt = np.exp(-time_gap / tau)
            # exp_wgt = 1 / time_gap
            edges_from_this_neuron = list(
                # nid_from, t_from, nid_to, t_to, t_gap, exp_wgt
                zip(repeat(n), repeat(t), nids[idx], ts[idx], time_gap, exp_wgt))
            EDGES += edges_from_this_neuron

    # make a weighted adjacency matrix
    D = np.zeros(shape=(xx.shape[0], xx.shape[0]))
    for e in EDGES:
        D[e[2], e[0]] += e[5]

    edge_index = []
    edge_weights = []
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if D[i, j] > 0:
                edge_index.append([j, i])  # 2 x T array (top row: from, bottom row: to)
                edge_weights.append(D[i, j])
    edge_index = torch.tensor(np.vstack(edge_index).T, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(np.unique(edge_index), dtype=torch.long).view(-1, 1)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

    mapdict = {it.item(): i for i, it in enumerate(data.x)}

    ei = torch.zeros_like(data.edge_index)
    ei[0, :] = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.edge_index[0, :])))
    ei[1, :] = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.edge_index[1, :])))
    d = torch.zeros_like(data.x)
    d = torch.LongTensor(list(map(lambda x: mapdict[x.item()], data.x)))

    tmp = data.edge_index.clone()
    data.edge_index = ei.clone()
    ei = tmp

    # return data, D, ei, d, EDGES
    return data