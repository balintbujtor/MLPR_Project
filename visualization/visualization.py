import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluation import computeConfusionMatrix, computeDCF, computeMinDCF, computeNormalisedDCF

def plotHistogram(data1, data2):
    
    plt.hist(data1, bins=25, density=True, fc=(0, 0, 1, 0.5))
    plt.hist(data2, bins=25, density=True, fc=(1, 0, 0, 0.5))
    plt.legend(["class false", "class true"])
    plt.show()
    
    return

def plotScatter(class0, class1):
    plt.scatter(class0[0, :], class0[1, :], fc=(0, 0, 1, 0.5))
    plt.scatter(class1[0, :], class1[1, :], fc=(1, 0, 0, 0.5))
    plt.legend(["class false", "class true"])
    plt.xlabel("attribute 0")
    plt.ylabel("attribute 1")
    plt.show()
    
    return

def plotCorrMat(data):
        # TODO: make it correct and contain info
    corrMat = np.corrcoef(data)
    plt.matshow(corrMat)
    plt.show()


def plotCumVarRatios(ratios, dims):
    # Plot the fraction of explained variance 
    plt.plot(range(0, dims), ratios) 
    plt.xlabel('Number of Principal Components') 
    plt.ylabel('Cumulative Variance Ratio') 
    plt.title('Fraction of Explained Variance with PCAs') 
    plt.grid(True) 
    plt.show()


def plotROCCurve(scores : np.ndarray, labels: np.ndarray):

    FPRs = []
    TPRs = []
    sorted_scores = np.sort(scores)
    for threshold in sorted_scores:
        comparisonResult = scores > threshold
        predictedLabels = np.where(comparisonResult, 1, 0)     
    
        confMatrix = computeConfusionMatrix(predictedLabels, labels)
        FNR = confMatrix[0][1] / (confMatrix[0][1] + confMatrix[1][1])
        FPR = confMatrix[1][0] / (confMatrix[1][0] + confMatrix[0][0])
        TPR = 1 - FNR
        FPRs.append(FPR)
        TPRs.append(TPR)
    
    plt.figure()
    plt.plot(FPRs, TPRs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()


def plotBayesError(effPriorLogOdds: np.ndarray, scores : np.ndarray, labels: np.ndarray):
    
    normDCFs = []
    minDCFs = []
    for priorlog in effPriorLogOdds:
        
        piTilde = 1 / (1 + np.exp(-priorlog))
        threshold = - np.log((piTilde * 1) / ((1 - piTilde)*1))
        
        comparisonResult = scores > threshold
        predictedLabels = np.where(comparisonResult, 1, 0)
        
        confMat = computeConfusionMatrix(predictedLabels, labels)
        DCF = computeDCF(confMat, piTilde, 1, 1)
        normDCF = computeNormalisedDCF(DCF, piTilde, 1, 1)
        minDCF = computeMinDCF(scores, labels, piTilde, 1, 1)
        
        normDCFs.append(normDCF)
        minDCFs.append(minDCF)
    
    return normDCFs, minDCFs