#!/usr/bin/env python3
"""
module 10-kmeans
contains function kmeans
"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset"""
    k_model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = k_model.labels_
    C = k_model.cluster_centers_

    return C, clss
