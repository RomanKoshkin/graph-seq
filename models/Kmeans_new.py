import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
import sys
from termcolor import cprint

sys.path.append('../')
from utils.utils import tonp


class Kmeans(MiniBatchKMeans):

    def __init__(self, params, debug=False):
        super().__init__()
        cprint("using sklearn partial fit", color='magenta')
        # self.kmeans = MiniBatchKMeans()
        self.n_clusters = params.K
        self.batch_size = params.Ts
        self.max_iter = 1
        self.device = params.device
        self.z_dim = params.z_dim
        self.debug = debug

    def fits(self, x):
        self.dat__ = tonp(x.T)
        self.partial_fit(self.dat__)

    @property
    def cluster_ids(self):
        return torch.tensor(self.predict(self.dat__), dtype=torch.long, device=self.device)

    @property
    def centroids(self):
        # NOTE: each row is a centroid
        return torch.tensor(self.cluster_centers_, dtype=torch.float, device=self.device)
