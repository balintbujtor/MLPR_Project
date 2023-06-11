import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def computeConfusionMatrix(predictedLabels: np.ndarray, correctLabels: np.ndarray):
    nClasses = np.max(correctLabels) - np.min(correctLabels) + 1
    
    confusionMatrix = np.zeros((nClasses, nClasses), dtype=int)
    
    for i in range(len(correctLabels)):
        confusionMatrix[predictedLabels[i]][correctLabels[i]] += 1
    
    return confusionMatrix


def computeOptimalBayesDecisionBinary(prior: np.ndarray, cfn: float, cfp: float, llr: np.ndarray):
    
    threshold = - np.log((prior * cfn) / ((1 - prior)*cfp))
    
    comparisonResult = llr > threshold
    predictedLabels = np.where(comparisonResult, 1, 0)
        
    return predictedLabels


def computeDCF(confusionMatrix: np.ndarray, prior: float, cfn: float, cfp: float):
    
    FNR = confusionMatrix[0][1] / (confusionMatrix[0][1] + confusionMatrix[1][1])
    FPR = confusionMatrix[1][0] / (confusionMatrix[1][0] + confusionMatrix[0][0])
    
    BayesRisk = prior*cfn*FNR + (1 - prior)*cfp*FPR
    
    return BayesRisk

def computeNormalisedDCF(BayesRisk : np.ndarray, prior: float, cfn: float, cfp: float):

    dummyBayes = min(prior*cfn, (1 - prior)*cfp)
    
    return BayesRisk / dummyBayes


def computeMinDCF(scores : np.ndarray, labels: np.ndarray, prior: float, cfn: float, cfp: float):
    
    normDCFs = []
    sorted_scores = np.sort(scores)
    for threshold in sorted_scores:
        comparisonResult = scores > threshold
        predictedLabels = np.where(comparisonResult, 1, 0)     
    
        confMatrix = computeConfusionMatrix(predictedLabels, labels)
        DCF = computeDCF(confMatrix, prior, cfn, cfp)
        normDCF = computeNormalisedDCF(DCF, prior, cfn, cfp)
        normDCFs.append(normDCF)
        
    return min(normDCFs)