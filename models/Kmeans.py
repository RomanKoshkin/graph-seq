import torch
import torch.nn.functional as F


class Kmeans(object):

    def __init__(self, params, debug=False):
        self.K = params.K
        self.device = params.device
        self.z_dim = params.z_dim
        self.centroids_initialized = False
        self.T = 0
        self.debug = debug
        if self.debug:
            self.C = []

    def _init_centroids(self, x):
        # https://www.kdnuggets.com/2020/06/centroid-initialization-k-means-clustering.html
        ds = x.T
        centroids = [ds[0]]
        for _ in range(1, self.K):
            dist_sq = torch.tensor([min([torch.inner(c - x, c - x) for c in centroids]) for x in ds])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum(dim=-1)
            r = torch.rand(size=(1,))

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(ds[i])

        # self.centroids = torch.rand(size=(self.K, self.z_dim)).to(self.device)
        # maxima = x.max(dim=1)[0].to(self.device)
        # minima = x.min(dim=1)[0].to(self.device)
        # self.centroids = self.centroids * (maxima - minima) + minima
        self.centroids = torch.vstack(centroids)
        self.centroids_initialized = True
        self.T = x.shape[1]
        if self.debug:
            self.C.append(self.centroids.cpu().detach().numpy())

    def step(self, x):
        if not self.centroids_initialized:
            self._init_centroids(x)
        distances = torch.cdist(x.T, self.centroids, p=2)
        assert (distances == distances).any(), "nans in distances"

        self.cluster_ids = distances.argmin(dim=1)
        self.soft_cluster_ids = F.softmin(distances, dim=1)
        self.update_centroids(x)
        if self.debug:
            self.C.append(self.centroids.cpu().detach().numpy())

    def update_centroids(self, x):
        centroids = torch.zeros_like(self.centroids)  # NOTE: to prevent updating centroids in-place

        for k in range(self.K):

            selects = x[:, self.cluster_ids == k]
            if selects.size()[1] == 0:
                self._init_centroids(x)
                print('reinit centroids')
            centroids[k, :] = selects.mean(dim=1)

            assert (self.centroids == self.centroids).any(), "nans in centroids"
        self.centroids = centroids

    def fit(self, x, steps=10):
        for i in range(steps):
            self.step(x.squeeze())

    def get_ids(self):
        return self.cluster_ids, self.soft_cluster_ids


# params.K = 2
# kmeans = Kmeans(params, debug=True)
# kmeans.fit(x)