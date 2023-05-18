import numpy as np
import random

def vcol(array):
    return array.reshape((array.shape[0], 1))


def vrow(array):

    return array.reshape((1, array.shape[0]))

def ML_estimates(XND : np.ndarray) :
    """returns the estimated mu and cov

    Args:
        XND (np.ndarray): dataset
    """
    
    mu_ML = vcol(XND.sum(1) / XND.shape[1])
    cov_ML = np.dot((XND - mu_ML), (XND - mu_ML).T)/XND.shape[1]
    return mu_ML, cov_ML