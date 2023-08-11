import numpy as np
import scipy
from helpers import helpers

def logMVG(traindata: np.ndarray, trainlabel: np.ndarray, testdata: np.ndarray, nclasses: int, prior : np.ndarray) -> tuple[np.ndarray, np.ndarray]:   
    """
    Multivariate Gaussian Classifier that computes the posterior probability
    for a given dataset and prior probability.

    This implementation is a more robust one of MVG that uses logs
    to avoid numeric issues.

    Args:
        traindata (np.ndarray): data on which to train the model
        trainlabel (np.ndarray): labels for the training set
        testdata (np.ndarray): data on which to evaluate the model
        nclasses (int): number of classes in the dataset
        prior (np.ndarray): prior probabilities in the dataset

    Returns:
        tuple[np.ndarray, np.ndarray]: returns the Joint probability and the posterior probability
    """
    logscore = []
    
    for i in range(0, nclasses):
        trainClassSample = traindata[:, trainlabel==i]

        mu, cov = helpers.ML_estimates(trainClassSample) 
        logpdf = helpers.logpdf_GAU_ND(testdata, mu, cov)
        
        logscore.append(helpers.vrow(logpdf))
        
    logscore = np.vstack(logscore)
    
    logSJoint = logscore + np.log(prior)
    logSMarginal = helpers.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    
    return np.exp(logSJoint), np.exp(logSPost)    


def logNaiveBayes(traindata: np.ndarray, trainlabel: np.ndarray, testdata: np.ndarray, nclasses: int, prior : np.ndarray) -> tuple[np.ndarray, np.ndarray]:   
    """
    Naive Bayes Gaussian Classifier

    Args:
        traindata (np.ndarray): data on which to train the model
        trainlabel (np.ndarray): labels for the training set
        testdata (np.ndarray): data on which to evaluate the model
        nclasses (int): number of classes in the dataset
        prior (np.ndarray): prior probabilities in the dataset

    Returns:
        tuple[np.ndarray, np.ndarray]: returns the Joint probability and the posterior probability
    """
    
    
    logscore = []
    for i in range(0, nclasses):
        trainClassSample = traindata[:, trainlabel==i]

        mu, cov = helpers.ML_estimates(trainClassSample)
        id = np.identity(cov.shape[1])
        naivecov = cov * id
        logpdf = helpers.logpdf_GAU_ND(testdata, mu, naivecov)
        logscore.append(helpers.vrow(logpdf))
        
    logscore = np.vstack(logscore)
    
    logSJoint = logscore + np.log(prior)
    logSMarginal = helpers.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    return np.exp(logSJoint), np.exp(logSPost)


def logTiedMVG(traindata: np.ndarray, trainlabel: np.ndarray, testdata: np.ndarray, nclasses: int, prior : np.ndarray) -> tuple[np.ndarray, np.ndarray]:   
    """
    Computes the the tied Multivariate Gaussian Density for a given dataset
    Tied: only one global covariance matrix that is summed and averaged of the class covs
    Log: more robust approach
    Args:
        traindata (np.ndarray): data on which to train the model
        trainlabel (np.ndarray): labels for the training set
        testdata (np.ndarray): data on which to evaluate the model
        nclasses (int): number of classes in the dataset
        prior (np.ndarray): prior probabilities in the dataset

    Returns:
        tuple[np.ndarray, np.ndarray]: returns the Joint probability and the posterior probability
    """
    logscore = []    
    mus = []
    tied_cov= np.zeros((traindata.shape[0], traindata.shape[0]), dtype = float)

    for i in range(0, nclasses):
        trainClassSample = traindata[:, trainlabel==i]

        mu, cov = helpers.ML_estimates(trainClassSample)
        mus.append(mu)
        tied_cov += cov*trainClassSample.shape[1]

    tied_cov /= traindata.shape[1]

    for i in range(0, nclasses):
        logpdf = helpers.logpdf_GAU_ND(testdata, mus[i], tied_cov)

        logscore.append(helpers.vrow(logpdf))

    logscore = np.vstack(logscore)
    logSJoint = logscore + np.log(prior)
    logSMarginal = helpers.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    return np.exp(logSJoint), np.exp(logSPost)


def logTiedNaiveBayes(traindata: np.ndarray, trainlabel: np.ndarray, testdata: np.ndarray, nclasses: int, prior : np.ndarray) -> tuple[np.ndarray, np.ndarray]:  
    """
    Tied Naive Bayes Gaussian Classifier

    Args:
        traindata (np.ndarray): data on which to train the model
        trainlabel (np.ndarray): labels for the training set
        testdata (np.ndarray): data on which to evaluate the model
        nclasses (int): number of classes in the dataset
        prior (np.ndarray): prior probabilities in the dataset

    Returns:
        tuple[np.ndarray, np.ndarray]: returns the Joint probability and the posterior probability
    """

    logscore = []    
    mus = []
    tied_cov= np.zeros((traindata.shape[0], traindata.shape[0]), dtype = float)

    for i in range(0, nclasses):
        trainClassSample = traindata[:, trainlabel==i]
        
        mu, cov = helpers.ML_estimates(trainClassSample)
        id = np.identity(cov.shape[1])
        naivecov = cov * id
        mus.append(mu)
        tied_cov += naivecov*trainClassSample.shape[1]
    
    tied_cov /= traindata.shape[1]
    
    for i in range(0, nclasses):        
        logpdf = helpers.logpdf_GAU_ND(testdata, mus[i], tied_cov)
        logscore.append(helpers.vrow(logpdf))
        
    logscore = np.vstack(logscore)
    logSJoint = logscore + np.log(prior)
    logSMarginal = helpers.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    
    return np.exp(logSJoint), np.exp(logSPost)
