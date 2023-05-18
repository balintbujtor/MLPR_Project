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


def splitToKFold(nsamples : int, nfolds=10, LOO=False) -> tuple[int, np.ndarray]:
    """
    Splits the dataset based on the number of samples into nfolds (almost) equal parts
    
    If LOO is true, then K-Folds becomes Leave-One-Out

    Args:
        nsamples (int): number of samples in the dataset
        nfolds (int, optional): number of folds, the function split the data into nfolds parts. Defaults to 10.
        LOO (bool, optional): if True, nfolds = nsamples --> LOO method. Defaults to False.

    Returns:
        tuple[int, np.ndarray]: 
            the number of splits in the dataset (iterations to perform)
            the bounds of the indeces for the split
    """
    if(LOO == True):
        nFolds = nsamples
    elif(nfolds > nsamples):
        nFolds = nsamples
    else:
        nFolds = nfolds
    
    foldBounds = np.full(nFolds, nsamples // nFolds, dtype=int)
    foldBounds[: nsamples % nFolds] += 1
        
    for i in range(1, nFolds):
        foldBounds[i] += foldBounds[i - 1]
    
    nsplits = foldBounds.size
    return  nsplits, foldBounds


def getCurrentKFoldSplit(data : np.ndarray, labels : np.ndarray, iter : np.ndarray, foldBounds : np.ndarray):
    """
    Splits the dataset based on the iteration and the foldbounds into traint and test

    Args:
        data (np.ndarray): dataset
        labels (np.ndarray): labels of the dataset
        iter (np.ndarray): number of K-fold iteration
        foldBounds (np.ndarray): bounds of indeces for the split

    Returns:
        the split up train and test set
    """
    if(iter == 0):
        lowerbound = 0
        upperbound = foldBounds[iter]

        testdata = data[:, lowerbound : upperbound]
        testlabels = labels[ lowerbound : upperbound]
        
        traindata =data[:, upperbound :]
        trainlabels =labels[upperbound :]
    else:
        lowerbound = foldBounds[iter - 1]
        upperbound = foldBounds[iter]

        testdata = data[:, lowerbound : upperbound]
        testlabels = labels[ lowerbound : upperbound]
        
        traindata = np.concatenate((data[:, : lowerbound], data[:, upperbound :]), axis=1)
        trainlabels =np.concatenate((labels[ : lowerbound], labels[upperbound :]))
    
    return (traindata, trainlabels), (testdata, testlabels) 

def compute_accuracy(predictedLabels : np.ndarray, correctLabels : np.ndarray) -> float:
    """    
    Compute the accuracy of a given classifier,
    based on the predicted and the correct labels.

    Args:
        predictedLabels (np.ndarray): the predictions of the classifier
        correctLabels (np.ndarray): the correct labels

    Returns:
        float: accuracy
    """
    correctPredictions = np.sum(predictedLabels == correctLabels)
    
    accuracy = correctPredictions / correctLabels.size
    
    return accuracy