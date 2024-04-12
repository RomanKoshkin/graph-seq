import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


def get_graph_stats(logits: torch.Tensor, params, fast=False):
    preds = logits.argmax(dim=1).tolist()
    Z = np.zeros((params.K, params.K))
    for i in range(len(preds) - 1):
        Z[preds[i + 1], preds[i]] += 1
    Z /= Z.sum()

    if not fast:
        DG = nx.from_numpy_array(Z.T, create_using=nx.DiGraph)  # takes from i to j, so we transpose
        return Z, nx.average_shortest_path_length(DG), nx.clustering(DG)
    else:
        return Z, None, None


def plotWeightedGraph(D):

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    DG = nx.from_numpy_array(D.T, create_using=nx.DiGraph)  # takes from i to j, so we transpose
    # nx.draw(DG, with_labels=True) # you can just call `nx.draw() on you graph, if you just need to quickly plot it

    # define node positions
    pos = nx.spring_layout(DG, seed=3113794652)  # positions for all nodes

    # draw the edges
    _ = nx.draw_networkx_edges(DG,
                               pos,
                               width=1,
                               alpha=0.5,
                               edge_color="tab:blue",
                               connectionstyle="arc3,rad=0.12",
                               ax=ax[0])

    # draw the nodes
    _ = nx.draw_networkx_nodes(DG, pos, node_color="red", alpha=0.3, ax=ax[0])

    # add node labels
    labels = {i: str(i) for i in list(DG.nodes)}
    _ = nx.draw_networkx_labels(DG, pos, labels, font_size=22, font_color="black", ax=ax[0])

    # add edge labels
    labels = {e: f"{e[0]}-{e[1]}:{DG.get_edge_data(*e)['weight']}" for e in list(DG.edges)}
    _ = nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels, ax=ax[0])

    print(D)
    ax[1].imshow(D)