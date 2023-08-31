import numpy as np
from nearest_neighbors import KNNClassifier


def kfold(n, n_folds=5):
    index = 0
    mask = np.zeros(n, dtype=bool)
    indices = np.arange(n)
    size = (n // n_folds) + 1
    splits = []
    for _ in range(n % n_folds):
        mask[index:index + size] = True
        test = indices[mask]
        train = indices[np.logical_not(mask)]
        mask[index:index + size] = False
        index += size
        splits.append((train, test))
    size = n // n_folds
    for _ in range(n_folds - n % n_folds):
        mask[index:index + size] = True
        test = indices[mask]
        train = indices[np.logical_not(mask)]
        mask[index:index + size] = False
        index += size
        splits.append((train, test))
    return splits


def knn_cross_val_score(X, y, k_list, cv=None, score='accuracy', **kwargs):
    eps = 1E-5
    if cv is None:
        cv = kfold(n=len(X))
    k_score = dict.fromkeys(k_list)
    for k in k_score:
        k_score[k] = np.empty(len(cv))
    for index, [train, test] in enumerate(cv):
        numb_cl = KNNClassifier(k=k_list[-1], **kwargs)
        numb_cl.fit(X[train], y[train])
        return_distance = kwargs['weights']
        classes = np.unique(y[train])
        if return_distance:
            dist, neigh = numb_cl.find_kneighbors(
                X[test], return_distance=True)
            for k in k_list:
                weights = 1 / (eps + dist[:, :k])
                mask = classes[None, None, :] == \
                    y[train][neigh[:, :k]].T[:, :, None]
                prediction = classes[np.argmax(
                    np.sum(weights.T[:, :, None] * mask, axis=0), axis=1)]
                k_score[k][index] = (np.sum(prediction == y[test]) /
                                     len(y[test]))
        else:
            neigh = numb_cl.find_kneighbors(
                X[test], return_distance=False)
            for k in k_list:
                mask = classes[None, None, :] == \
                    y[train][neigh[:, :k]].T[:, :, None]
                prediction = classes[np.argmax(
                    np.sum(mask, axis=0), axis=1)]
                k_score[k][index] = (np.sum(prediction == y[test]) /
                                     len(y[test]))
    return k_score


def knn_cross_val_score_aug(X, y, k_list, *args, cv=None, score='accuracy',
                            aug=None, **kwargs):
    eps = 1E-5
    if cv is None:
        cv = kfold(n=len(X))
    k_score = dict.fromkeys(k_list)
    for k in k_score:
        k_score[k] = np.empty(len(cv))
    for index, [train, test] in enumerate(cv):
        X_train_aug = np.empty(X[train].shape)
        for idx, image in enumerate(X[train]):
            aug_image = aug(image.reshape((28, 28)), *args)
            X_train_aug[idx] = np.ravel(aug_image)
        numb_cl = KNNClassifier(k=k_list[-1], **kwargs)
        numb_cl.fit(np.vstack((X[train], X_train_aug)),
                    np.hstack((y[train], y[train])))
        return_distance = kwargs['weights']
        classes = np.unique(y[train])
        if return_distance:
            dist, neigh = numb_cl.find_kneighbors(
                X[test], return_distance=True)
            for k in k_list:
                weights = 1 / (eps + dist[:, :k])
                mask = classes[None, None, :] == \
                    np.hstack((y[train], y[train]))[neigh[:, :k]].T[:, :, None]
                prediction = classes[np.argmax(
                    np.sum(weights.T[:, :, None] * mask, axis=0), axis=1)]
                k_score[k][index] = (np.sum(prediction == y[test]) /
                                     len(y[test]))
        else:
            neigh = numb_cl.find_kneighbors(
                X[test], return_distance=False)
            for k in k_list:
                mask = classes[None, None, :] == \
                    np.hstack((y[train], y[train]))[neigh[:, :k]].T[:, :, None]
                prediction = classes[np.argmax(
                    np.sum(mask, axis=0), axis=1)]
                k_score[k][index] = (np.sum(prediction == y[test]) /
                                     len(y[test]))
    return k_score
