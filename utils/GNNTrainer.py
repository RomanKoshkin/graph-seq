import matplotlib.pyplot as plt

from tqdm import trange
from termcolor import cprint
import numpy as np
import torch, copy
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.distributions import Categorical

from .constants import bar_format
from .utils import AttributeDict, tonp
from .plotting import Plot
from .losses import CentroidProximityLoss, DispersionLoss, L2_loss, DimensionSimilarityLoss, SelfSimilarityLoss


class LossDict(AttributeDict):

    def __init__(self):
        super().__init__()
        self.PROXIMITY = []
        self.VAR = []
        self.ENTROPY = []
        self.XENT = []
        self.DISPERS = []
        self.L2 = []
        self.TV = []
        self.JUMPINESS = []
        self.DIVERSITY = []
        self.DISPERSION = []
        self.SELFSIM = []
        self.NL = []

    def mean(self):
        for k, v in self.items():
            self[k] = np.mean(v) if len(v) > 0 else 0


class Trainer:

    def __init__(self, DATA, X_, model, optimizer, scheduler, kmeans, params, step_sz):
        self.params = params
        self.step_sz = step_sz
        self.X_ = X_
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        if self.scheduler is None:
            cprint("Not using lrate scheduling", color='blue')
        self.kmeans = kmeans
        self.losses = []

        self.LIM = (5000, 15000)
        self.EP = 0
        self.MAX_EPOCHS = 40000

        # Xent = nn.CrossEntropyLoss(reduction='mean', weight=1.0/counts)
        self.Xent = nn.CrossEntropyLoss(reduction='mean')

        self.loader = DataLoader(DATA, batch_size=len(DATA), shuffle=False)  # TODO: move to GPU
        # self.loader = DataLoader(DATAnull, batch_size=len(DATAnull), shuffle=False) # TODO: move to GPU

        self.lossdict = LossDict()
        self.writer = SummaryWriter(log_dir='../runs/')

    def train(self):
        pbar = trange(self.EP, self.MAX_EPOCHS, bar_format=bar_format)
        for j in pbar:
            try:
                self.lossdict = LossDict()
                for batch in self.loader:

                    self.optimizer.zero_grad()
                    embeds, logits = self.model(batch.x,
                                                batch.edge_index,
                                                batch.edge_weight,
                                                batch.batch,
                                                deterministic=False)

                    # cluster, get labels
                    self.kmeans.fit(embeds.T)  # expects samples to be in rows, features in columns (unlike sklearn)
                    # kmeans.fits(embeds.T) # expects samples to be in rows, features in columns (unlike sklearn)
                    ids = self.kmeans.cluster_ids
                    self.ids = ids

                    tv_loss = embeds.diff(dim=0).pow(2).mean()  # encourage temporal consistency or cluster assignments
                    centroid_proximity_loss = -CentroidProximityLoss(self.kmeans.centroids)  # push centroids apart
                    var_loss = torch.tensor([0.0], device=self.params.device)  #kmeans.soft_cluster_ids.var(dim=0).sum()
                    l2_loss = L2_loss(self.model, self.params)  # smoothen model parameters
                    dispersion_loss = DispersionLoss(embeds, self.kmeans, self.params)  # ???
                    class_probabilities = F.softmax(logits, dim=1).mean(dim=0)
                    class_diversity_loss = -Categorical(probs=class_probabilities).entropy()  # class diversity loss
                    dim_sim_loss = DimensionSimilarityLoss(logits.T, self.params)
                    jumpiness_loss = F.gumbel_softmax(
                        logits,
                        dim=1,
                        tau=0.1,
                        hard=True,
                    ).diff(dim=0).abs().mean()  # temporal consistency loss via jumpiness of labels (TV does the same)
                    selfsim_loss = -SelfSimilarityLoss(embeds)  # bullshit: self-similarity of embeddings
                    nl = torch.cdist(
                        embeds.unsqueeze(1),
                        self.kmeans.centroids[ids, :].unsqueeze(1),
                    ).squeeze().pow(2).mean()

                    self.ProximityToNearestCentroid = torch.cdist(
                        self.kmeans.centroids.unsqueeze(0).expand(embeds.shape[0], -1, -1),
                        embeds.unsqueeze(1).detach(),
                        p=2,
                    ).squeeze().min(dim=-1)[0].reciprocal().detach()

                    # xent_loss = self.Xent(embeds, ids)
                    # NOTE: WIP
                    self.Xent = nn.CrossEntropyLoss(reduction='none')
                    xent_loss = (self.Xent(
                        embeds,
                        ids,
                    ) * self.ProximityToNearestCentroid).mean()

                    loss = xent_loss + 1.0 * tv_loss * 0.1 * l2_loss
                    # loss = xent_loss + 0.1*tv_loss + 0.01*centroid_proximity_loss

                    loss.backward(retain_graph=False)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.lossdict.NL.append(nl.item())
                    self.lossdict.PROXIMITY.append(centroid_proximity_loss.item())
                    self.lossdict.VAR.append(var_loss.item())
                    # self.lossdict.ENTROPY.append(entropy_loss.item())
                    self.lossdict.XENT.append(xent_loss.item())
                    self.lossdict.DISPERSION.append(dispersion_loss.item())
                    self.lossdict.L2.append(l2_loss.item())
                    self.lossdict.TV.append(tv_loss.item())
                    self.lossdict.JUMPINESS.append(F.softmax(embeds, dim=1).diff(dim=0).abs().mean().item())
                    self.lossdict.DIVERSITY.append(class_diversity_loss.squeeze().item())
                    self.lossdict.SELFSIM.append(selfsim_loss.squeeze().item())

                embeds_np = tonp(embeds)
                self.embeds_np = embeds_np
                self.logits = logits
                self.lossdict.mean()
                self.losses.append(copy.deepcopy(dict(self.lossdict)))

                pbar.set_description("".join([f'{k}: {v:.2f} | ' for k, v in self.lossdict.items()]))

                for k, v in self.lossdict.items():
                    self.writer.add_scalar(f"loss/{k}", v, self.EP)

                # dump snapshots
                if self.EP % 20 == 0:
                    # LIM = (5000, 18000)
                    LIM = (0, 6000)
                    Plot(self.embeds_np, self.X_, self.ids, self.step_sz, self.logits, self.params, self.LIM, self.EP)
                    plt.savefig(f'../data/proj_{self.EP:06}.png')
                    plt.close('all')

                self.EP += 1

            except KeyboardInterrupt:
                print('User interrupt')
                self.writer.flush()

                # LIM = (0, 3000)

                ax = Plot(self.embeds_np, self.X_, self.ids, self.step_sz, self.logits, self.params, self.LIM)
                torch.cuda.empty_cache()
                break

    def plot(self, LIM=None):
        if LIM is not None:
            lim = LIM
        else:
            lim = self.LIM
        return Plot(self.embeds_np, self.X_, self.ids, self.step_sz, self.logits, self.params, lim)