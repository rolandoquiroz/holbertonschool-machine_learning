#!/usr/bin/env python3
"""
module 11-gmm
contains function gmm
"""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset"""
    gaussian_mixture = sklearn.mixture.GaussianMixture(n_components=k)
    gaussian_mixture.fit(X)
    pi = gaussian_mixture.weights_
    m = gaussian_mixture.means_
    S = gaussian_mixture.covariances_
    clss = gaussian_mixture.predict(X)
    bic = gaussian_mixture.bic(X)
    return pi, m, S, clss, bic
