from scipy.stats import hmean, gmean
import numpy as np


def mean_path(paths):
    return sum(paths) / len(paths)


def median_path(paths):
    return np.median(paths, axis=1)


def harmonic_mean_path(paths):
    return hmean(paths)


def geometric_mean_path(paths):
    return gmean(paths)


def weighted_average_path(paths, weights):
    return sum(p * w for p, w in zip(paths, weights)) / sum(weights)
