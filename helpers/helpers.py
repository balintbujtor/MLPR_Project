import numpy as np
import random

def vcol(array):
    return array.reshape((array.shape[0], 1))


def vrow(array):

    return array.reshape((1, array.shape[0]))

def logpdf_GAU_ND(x : np.ndarray, mu :np.ndarray, C :np.ndarray) -> np.ndarray:
    """computes the log density of the dataset x given mu and C
    
    M is the size of the feature vector

    Args:
        x (np.ndarray): dataset
        mu (np.ndarray): mean that is of shape (M, 1)
        C (np.ndarray): covariance matrix that is of shape (M, M)

    Returns:
        np.ndarray: log density
    """

    constant = 0.5*x.shape[0]*np.log(2*np.pi)
    logdetc = 0.5 * np.linalg.slogdet(C)[1]
    
    xc = x - vcol(mu)
    first_minor = np.dot(np.linalg.inv(C), xc) # cov*centered data, in case of iris 4x150
    v = xc*first_minor # element-wise multiplication (as in vector by vector), in case of iris yields 4x150
    final = 0.5 * v.sum(0) # sum over columns --> in case of iris yields 1x150

    return np.asarray(- constant - logdetc - final, dtype=float)


def ML_estimates(XND : np.ndarray) :
    """returns the estimated mu and cov

    Args:
        XND (np.ndarray): dataset
    """
    
    mu_ML = vcol(XND.sum(1) / XND.shape[1])
    cov_ML = np.dot((XND - mu_ML), (XND - mu_ML).T)/XND.shape[1]
    return mu_ML, cov_ML


def splitToKFold(D: np.ndarray, L: np.ndarray, K: int = 5, seed=0):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    
    for i in range(K):
        folds.append(D[:, idx[ (i*numberOfSamplesInFold) : ((i+1)*(numberOfSamplesInFold)) ]])
        labels.append(L[ idx[ (i*numberOfSamplesInFold) : ((i+1)*(numberOfSamplesInFold)) ]])
        
    return folds, labels

def getCurrentKFoldSplit(splitdata : np.ndarray, splitlabels : np.ndarray, curFold : int, nFolds: int = 5):
    
    trainingData = []
    trainingLabels = []
    evalData = []
    evalLabels = []
    
    for j in range(nFolds):
        if j != curFold:
            trainingData.append(splitdata[j])
            trainingLabels.append(splitlabels[j])
        else:
            evalData = splitdata[curFold]
            evalLabels = splitlabels[curFold]
    
    trainingData = np.hstack(trainingData)
    trainingLabels = np.hstack(trainingLabels)
    
    return trainingData, trainingLabels, evalData, evalLabels


def saveMinDCF(name: str, minDCFArray: np.ndarray, prior: float, zNorm: bool):
        formattedPrior = "{:.2f}".format(prior)
        
        np.save(f"results/npy/minDCF{name}_prior{formattedPrior}_Znorm{zNorm}", minDCFArray)
        np.savetxt(f"results/txt/minDCF{name}_prior{formattedPrior}_Znorm{zNorm}", minDCFArray)
