import numpy as np
import scipy.special
import preprocessing.preprocessing as preproc
import generativeModels.generativeModels as generativeModels
import evaluation.evaluation as eval

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


def trainAllGenerativeClassifiers(startPCA: int, endPCA: int, DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, nFolds: int, znorm: bool) -> tuple[list, list, list, list]:
    
    minDCFMVGarray = []
    minDCFTiedGarray = []
    minDCFNBarray = []
    minDCFTiedNBArray = []
    
    prior, Cfn, Cfp = workingPoint
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    # to try out different pca directions
    for j in range(startPCA, endPCA, -1):
        
        if(j == 11):
            # without PCA
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)
            kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
        
        else:
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)
            reducedData, _ = preproc.computePCA(DTR, j)
            kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)

        sLogPostMVG = []
        sLogPostTiedG = []
        sLogPostNB = []
        sLogPostTiedNB = []
        
        correctEvalLabels = []

        for i in range(0, nFolds):
            
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
            
            # same for all, do it only once
            correctEvalLabels.append(evalLabels)
            
            # MVG with K-fold
            _, sPostLogMVG = generativeModels.logMVG(trainingData, trainingLabels, evalData, 2, priorProb)
            sLogPostMVG.append(sPostLogMVG)
            
            # tied G
            _, sPostLogTiedG = generativeModels.logTiedMVG(trainingData, trainingLabels, evalData, 2, priorProb)
            sLogPostTiedG.append(sPostLogTiedG)
            
            # naive
            _, sPostLogNB = generativeModels.logNaiveBayes(trainingData, trainingLabels, evalData, 2, priorProb)
            sLogPostNB.append(sPostLogNB)
                
            #naivetied - tied model is expected to perform worse than the untied
            _, sPostLogTiedNB = generativeModels.logTiedNaiveBayes(trainingData, trainingLabels, evalData, 2, priorProb)
            sLogPostTiedNB.append(sPostLogTiedNB)
        
        correctEvalLabels = np.hstack(correctEvalLabels)

        # eval of MVG
        sLogPostMVG = np.hstack(sLogPostMVG)
        llrMVG = np.log(sLogPostMVG[1] / sLogPostMVG[0])
        minDCFMVG = eval.computeMinDCF(llrMVG, correctEvalLabels, prior, Cfn, Cfp)
        minDCFMVGarray.append([int(j), minDCFMVG])
        
        # eval of tied MVG
        sLogPostTiedG = np.hstack(sLogPostTiedG)
        llrTiedG = np.log(sLogPostTiedG[1] / sLogPostTiedG[0])
        minDCFTiedG = eval.computeMinDCF(llrTiedG, correctEvalLabels, prior, Cfn, Cfp)
        minDCFTiedGarray.append([int(j), minDCFTiedG])
        
        # eval of NB
        sLogPostNB = np.hstack(sLogPostNB)
        llrNB = np.log(sLogPostNB[1] / sLogPostNB[0])
        minDCFNB = eval.computeMinDCF(llrNB, correctEvalLabels, prior, Cfn, Cfp)
        minDCFNBarray.append([int(j), minDCFNB])
        
        # eval of tied NB
        sLogPostTiedNB = np.hstack(sLogPostTiedNB)
        llrTiedNB = np.log(sLogPostTiedNB[1] / sLogPostTiedNB[0])
        minDCFTiedNB = eval.computeMinDCF(llrTiedNB, correctEvalLabels, prior, Cfn, Cfp)
        minDCFTiedNBArray.append([int(j), minDCFTiedNB])


    return minDCFMVGarray, minDCFTiedGarray, minDCFNBarray, minDCFTiedNBArray

