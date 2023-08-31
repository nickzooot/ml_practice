import unittest
import distances as dist
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from nearest_neighbors import KNNClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from cross_validation import knn_cross_val_score, kfold
import nearest_neighbors


class TestDistances(unittest.TestCase):

    def setUp(self):
        self.a = np.random.random((10, 6))
        self.b = np.random.random((12, 6))

    def test_euclidean_distance(self):
        my_real = dist.euclidean_distance(self.a, self.b)
        comp_with_real = distance.cdist(self.a, self.b, metric='euclidean')
        self.assertEqual(np.allclose(my_real, comp_with_real), True)
        self.assertEqual(my_real.shape, comp_with_real.shape)

    def test_cosine_distance(self):
        my_real = dist.cosine_distance(self.a, self.b)
        comp_with_real = distance.cdist(self.a, self.b, metric='cosine')
        self.assertEqual(np.allclose(my_real, comp_with_real), True)
        self.assertEqual(my_real.shape, comp_with_real.shape)


class TestNearestNeighbors(unittest.TestCase):

    obj_num = 1000
    feature_num = 500
    train_num = 700
    k = 14\


    @classmethod
    def setUpClass(cls):
        cls.data = np.random.randint(0, 30, (cls.obj_num, cls.feature_num))
        cls.targets = np.random.randint(0, 10, 1000)

    def test_own_nearest_neighbors_kneighbors(self):
        data = self.data[:self.train_num]
        target_data = self.data[self.train_num:]
        my_real = nearest_neighbors.OwnNearestNeighbors(metric='euclidean')
        comp_with_real = NearestNeighbors(metric='euclidean',
                                          algorithm='brute')
        my_real.fit(data)
        comp_with_real.fit(data)
        k_list = [1, 5, self.train_num]
        for k in k_list:
            dist_my, neigh_my = my_real.kneighbors(
                target_data, n_neighbors=k, return_distance=True
            )
            dist_comp_with, neigh_comp_with = comp_with_real.kneighbors(
                target_data, n_neighbors=k, return_distance=True
            )
            self.assertEqual(np.allclose(dist_my, dist_comp_with), True)
            mask1 = neigh_my == neigh_comp_with
            mask2 = np.zeros(neigh_my.shape, dtype=bool)
            mask2[neigh_my != neigh_comp_with] = np.allclose(
                dist_my, dist_comp_with
            )
            self.assertEqual(np.all(mask1 + mask2), True, msg=k)

    def test_find_kneighbors(self):
        data_train = self.data[:self.train_num]
        data_test = self.data[self.train_num:]
        numb_cl_my = KNNClassifier(k=self.k,
                                   metric='cosine',
                                   test_block_size=11)
        numb_cl = KNeighborsClassifier(algorithm='brute', metric='cosine')
        numb_cl_my.fit(data_train, self.targets[:self.train_num])
        numb_cl.fit(data_train, self.targets[:self.train_num])
        dist_my, neigh_my = numb_cl_my.find_kneighbors(
            data_test,  return_distance=True
        )
        _dist, _neigh = numb_cl.kneighbors(
            data_test, n_neighbors=self.k, return_distance=True
        )
        self.assertEqual(np.allclose(dist_my, _dist), True)
        mask1 = neigh_my == _neigh
        mask2 = np.zeros(neigh_my.shape, dtype=bool)
        mask2[neigh_my != _neigh] = np.allclose(
            dist_my, _dist
        )
        self.assertEqual(np.all(mask1 + mask2), True)

    def test_predict(self):
        data_train = self.data[:self.train_num]
        data_test = self.data[self.train_num:]
        numb_cl_my = KNNClassifier(k=self.k,
                                   metric='cosine',
                                   test_block_size=11)
        numb_cl = KNeighborsClassifier(
            n_neighbors=self.k, algorithm='brute', metric='cosine'
        )
        numb_cl_my.fit(data_train, self.targets[:self.train_num])
        numb_cl.fit(data_train, self.targets[:self.train_num])
        self.assertEqual(
            np.all(numb_cl.predict(data_test) ==
                   numb_cl_my.predict(data_test)), True)


class TestCrossValidation(unittest.TestCase):
    obj_num = 100
    feature_num = 50
    train_num = 70
    folds_num = 10
    k = 14

    @classmethod
    def setUpClass(cls):
        cls.data = np.random.randint(0, 30, (cls.obj_num, cls.feature_num))
        cls.targets = np.random.randint(0, 10, cls.obj_num)

    def test_kfold(self):
        _kfold = KFold(n_splits=self.folds_num).split(self.data)
        kfold_my = kfold(self.obj_num, self.folds_num)
        for index, [train, test] in enumerate(_kfold):
            self.assertEqual(np.all(kfold_my[index][0] == train), True)
            self.assertEqual(np.all(kfold_my[index][1] == test), True)
        self.assertEqual(index + 1, len(kfold_my))

    def test_knn_cross_val_score(self):
        cross_val_my = knn_cross_val_score(
            self.data,
            self.targets,
            [self.k],
            cv=kfold(len(self.data), self.folds_num),
            weights=False)
        cross_val = cross_val_score(
            KNeighborsClassifier(n_neighbors=self.k),
            self.data,
            self.targets,
            scoring='accuracy',
            cv=kfold(len(self.data), self.folds_num))
        self.assertEqual(np.allclose(cross_val_my[self.k], cross_val), True)


if __name__ == '__main__':
    unittest.main()
