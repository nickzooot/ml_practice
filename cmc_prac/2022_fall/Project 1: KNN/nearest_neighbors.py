import numpy as np
import distances
from sklearn.neighbors import NearestNeighbors


class OwnNearestNeighbors:
    def __init__(self, metric):
        self.X = None
        self.metric = metric

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, X, n_neighbors, return_distance):
        if self.metric == 'euclidean':
            distance = distances.euclidean_distance(X, self.X)
        else:
            distance = distances.cosine_distance(X, self.X)

        index_array = np.argpartition(
            distance, kth=range(n_neighbors), axis=-1)[:, :n_neighbors]

        if return_distance:
            return (
                np.take_along_axis(
                    distance, index_array, axis=-1
                ),
                index_array
            )
        else:
            return index_array


class KNNClassifier:
    eps = 1E-5

    def __init__(self,
                 k=5,
                 strategy='my_own',
                 metric='euclidean',
                 weights=False,
                 test_block_size=512):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.neigh = None
        self.classes = None
        self.targets = None

    def fit(self, X, y):
        if self.strategy != 'my_own':
            self.neigh = NearestNeighbors(
                algorithm=self.strategy, metric=self.metric).fit(X)
        else:
            self.neigh = OwnNearestNeighbors(metric=self.metric).fit(X)
        self.targets = y
        self.classes = np.unique(y)

    def find_kneighbors(self, X, return_distance: bool):
        """

        Parametrs:
        _________
        X -- selection of objects
        return_distance -- bool

        Returns:
        _______
        The method returns a tuple of two np.ndarray of size (X.shape[0], k).
        [i, j] the element of the first array must be equal to the distance
        from the i-th object to its j-th nearest neighbor. [i, j] the element
        of the second array must be equal to the index of the j-th nearest
        neighbor from the training sample for an object with index i.
        If return_distance=False, only the second of the specified arrays
        is returned.

        The method uses the search
        strategy specified in the strategy class parameter
        """
        neigh = np.empty((X.shape[0], self.k), dtype=int)
        dist = None
        if return_distance:
            dist = np.empty((X.shape[0], self.k))
        i = 0
        while i < X.shape[0]:
            if return_distance:
                (dist[i:i + self.test_block_size],
                 neigh[i:i + self.test_block_size]) = self.neigh.kneighbors(
                    X[i:i + self.test_block_size],
                    n_neighbors=self.k,
                    return_distance=True)
            else:
                neigh[i:i + self.test_block_size] = self.neigh.kneighbors(
                    X[i:i + self.test_block_size],
                    n_neighbors=self.k,
                    return_distance=False)
            i += self.test_block_size
        if return_distance:
            return dist, neigh
        else:
            return neigh

    def predict(self, X):
        if self.weights:
            dist, neigh = self.find_kneighbors(
                X, return_distance=self.weights)
            weights = 1 / (KNNClassifier.eps + dist)
            mask = self.classes[None, None, :] == \
                self.targets[neigh].T[:, :, None]
            return self.classes[np.argmax(
                np.sum(weights.T[:, :, None] * mask, axis=0), axis=1)]
        else:
            neigh = self.find_kneighbors(
                X, return_distance=self.weights)
            mask = self.classes[None, None, :] == \
                self.targets[neigh].T[:, :, None]
            return self.classes[np.argmax(
                np.sum(mask, axis=0), axis=1)]
