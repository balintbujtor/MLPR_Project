import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluation import computeConfusionMatrix, computeDCF, computeMinDCF, computeNormalisedDCF

def plotHistogram(data1, data2, title):
    
    plt.hist(data1, bins=25, density=True, fc=(0, 0, 1, 0.5))
    plt.hist(data2, bins=25, density=True, fc=(1, 0, 0, 0.5))
    plt.legend(["class 0", "class 1"])
    plt.savefig(f'figures/{title}.png')
    # plt.show()
    plt.clf()
    
    return

def plotScatter(class0, class1, title):
    plt.scatter(class0[0, :], class0[1, :], fc=(0, 0, 1, 0.5))
    plt.scatter(class1[0, :], class1[1, :], fc=(1, 0, 0, 0.5))
    plt.legend(["class 0", "class 1"])
    plt.xlabel("attribute 0")
    plt.ylabel("attribute 1")
    plt.savefig(f'figures/{title}.png')
    # plt.show()
    plt.clf()
    
    return

def plotCorrMat(data, filename):
    corrMat = np.corrcoef(data)
    plt.matshow(corrMat)
    plt.colorbar()
    plt.savefig(f'figures/{filename}.png')
    # plt.show()
    plt.clf()

def plotCumVarRatios(ratios, dims):
    # Plot the fraction of explained variance 
    plt.plot(range(0, dims), ratios) 
    plt.xlabel('Number of Principal Components') 
    plt.ylabel('Cumulative Variance Ratio') 
    plt.title('Fraction of Explained Variance with PCAs') 
    plt.grid(True) 
    plt.savefig(f'figures/CumulativeVarianceRatios.png')
    # plt.show()
    plt.clf()


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


def plotLogRegDCFs(modelsToShow: list, modelNames: list, title: str, xparam: str, PCAdims: int):
    plt.figure()
    plt.grid(True)
    
    lines = ["-","--","-.",":"]
    displayedLegend = []
    for m, model in enumerate(modelsToShow):
        
        color = iter(plt.cm.rainbow(np.linspace(0, 1, len(PCAdims))))

        for pcadim in PCAdims:
            arr = []
            for i in range(len(model)):
                if model[i][0] == pcadim:
                    arr.append([model[i][1], model[i][2]])
            
            # Extract x and y values for plotting
            x_values = [point[0] for point in arr]
            y_values = [point[1] for point in arr]

            # Plot the data
            c = next(color)
            plt.plot(x_values, y_values, linestyle=lines[m], color=c)
            plt.xlim(min(x_values), max(x_values))
            curDim = "RAW" if pcadim == 11 else pcadim
            curLegend = f'{modelNames[m]} - {curDim}'
            displayedLegend.append(curLegend)

    plt.legend(displayedLegend)
    
    plt.xscale('log')
    plt.xlabel(xparam)
    plt.ylabel('minDCF')
    plt.title(title)
    
    plt.savefig(f'results/img/{title}.png')
    plt.show()
    plt.clf()


def plotDifGammaDCFs(modelsToShow: list, modelNames: list, title: str, xparam: str, PCAdims: int):
    plt.figure()
    plt.grid(True)
    
    displayedLegend = []
    for m, model in enumerate(modelsToShow):
        
        for pcadim in PCAdims:
            arr = []
            for i in range(len(model)):
                if model[i][0] == pcadim:
                    arr.append([model[i][1], model[i][2]])
            
            # Extract x and y values for plotting
            x_values = [point[0] for point in arr]
            y_values = [point[1] for point in arr]

            # Plot the data
            plt.plot(x_values, y_values)
            plt.xlim(min(x_values), max(x_values))
            curDim = "RAW" if pcadim == 11 else pcadim
            curLegend = f'{modelNames[m]} - {curDim}'
            displayedLegend.append(curLegend)

    plt.legend(displayedLegend)
    
    plt.xscale('log')
    plt.xlabel(xparam)
    plt.ylabel('minDCF')
    plt.title(title)
    
    plt.savefig(f'results/img/{title}.png')
    plt.show()
    plt.clf()