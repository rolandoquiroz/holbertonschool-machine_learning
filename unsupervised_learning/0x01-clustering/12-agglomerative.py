#!/usr/bin/env python3
"""
module 12-agglomerative
contains function agglomerative
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    my_linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(my_linkage,
                                            t=dist,
                                            criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(my_linkage,
                                       color_threshold=dist)
    plt.show()

    return clss
