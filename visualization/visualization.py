import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

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


def computeDetPoints(llr, label: np.ndarray):
    threshold = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    FNRPoints = np.zeros(label.shape[0] + 2)
    FPRPoints = np.zeros(label.shape[0] + 2)
    for (idx, t) in enumerate(threshold):
        pred = 1 * (llr > t)
        FNR = 1 - (np.bitwise_and(pred == 1, label == 1).sum() / (label == 0).sum())
        FPR = np.bitwise_and(pred == 1, label == 0).sum() / (label == 1).sum()
        FNRPoints[idx] = FNR
        FPRPoints[idx] = FPR
    return FNRPoints, FPRPoints


def plotDET(llr: list, labels: np.ndarray, plotNames: list, colors, file_name: str):

    for (idx, scores) in enumerate(llr):
        fpr, tpr = computeDetPoints(scores, labels)
        plt.plot(fpr, tpr, color=colors[idx], label=plotNames[idx])

    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig("results/img/" + file_name)
    plt.close()


def computeBayesError(effPriorLogOdds: np.ndarray, scores : np.ndarray, labels: np.ndarray):
    
    normDCFs = []
    minDCFs = []
    piTildes = []
    for priorlog in effPriorLogOdds:
        
        piTilde = 1 / (1 + np.exp(-priorlog))
        threshold = - np.log((piTilde * 1) / ((1 - piTilde)*1))
        
        comparisonResult = scores > threshold
        predictedLabels = np.where(comparisonResult, 1, 0)
        
        confMat = computeConfusionMatrix(predictedLabels, labels)
        DCF = computeDCF(confMat, piTilde, 1, 1)
        normDCF = computeNormalisedDCF(DCF, piTilde, 1, 1)
        minDCF = computeMinDCF(scores, labels, piTilde, 1, 1)
        
        piTildes.append(piTilde)
        normDCFs.append(normDCF)
        minDCFs.append(minDCF)
    
    return piTildes, normDCFs, minDCFs


def plotBayesError(scores: np.ndarray, labels: np.ndarray, title: str, calScores: np.ndarray = None, calLabels: np.ndarray = None):
    effPriorLogOdds = np.linspace(-4, 4, 100)

    piTildes, actDCF, minDCF = computeBayesError(effPriorLogOdds, scores, labels)
    nonCalResults = np.array([piTildes, actDCF, minDCF]).T
    
    np.save(f"results/npy/calibration/{title}_DCFS", nonCalResults)
    np.savetxt(f"results/txt/calibration/{title}_DCFs", nonCalResults)
    
    if calScores is not None and calLabels is not None:
        calPiTildes, calActDCFs, calMinDCFs = computeBayesError(effPriorLogOdds, calScores, calLabels)
        calResults = np.array([calPiTildes, calActDCFs, calMinDCFs]).T
        
        np.save(f"results/npy/calibration/{title}_calDCFs", calResults)
        np.savetxt(f"results/txt/calibration/{title}_calDCFs", calResults)
        
    plt.figure()
    plt.plot(effPriorLogOdds, actDCF, label='DCF', color='r')
    
    if calScores is not None and calLabels is not None:
        plt.plot(effPriorLogOdds, calActDCFs, label='calibrated DCF', color='b')
        
    plt.plot(effPriorLogOdds, minDCF, label='min DCF', color='y', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.grid(True)
    plt.xlabel('log odds')
    plt.ylabel('DCFs')
    plt.title(title)
    plt.legend()
    plt.show()    
    plt.savefig(f'results/img/{title}.png')



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




def plotGMM_BarChart(data: np.ndarray, zNormData: np.ndarray, pcaDims: list, modelNames: list, title: str):
    
    # Setting the positions and width for the bars
    num_col = len(data + zNormData) - 1
    width = 0.95 / num_col
    
    cmap = cm.get_cmap('Blues_r', len(pcaDims))
    cmap2 = cm.get_cmap('Reds_r', len(pcaDims))
    
    fig, ax = plt.subplots(figsize=(16,10))

    compRange = np.unique([int(row[1]) for row in data])
    
    for  i, compNum in enumerate(compRange):
        
        color1 = iter(cmap(np.linspace(0.2, 0.5, len(pcaDims))))
        color2 = iter(cmap2(np.linspace(0.2, 0.5, len(pcaDims))))
        
        for d, pcaDim in enumerate(pcaDims):
            
            c1 = next(color1)
            c2 = next(color2)
            
            deltaP = width * d * 2
            
            y_values = [row[2] for row in data if row[0] == pcaDim and row[1] == compNum]
            zy_values = [row[2] for row in zNormData if row[0] == pcaDim and row[1] == compNum]
            
            curDim = "No PCA" if pcaDim == 11 else f'PCA {pcaDim}'

            # Calculate x positions for each group of bars
            x_pos = 1*i + deltaP
            x_pos2 = 1*i + deltaP + width
            # Normalize colors to use only the middle half of the colormap


            # Plot the bars for the first model
            ax.bar(x_pos, y_values, width=width, color=c1)
            ax.text(x_pos, y_values[0]+ 0.01, f'{curDim}', ha="center", va="bottom", rotation=90)
            
            # Plot the bars for the second model next to the first model's bars
            ax.bar(x_pos2, zy_values, width=width, color=c2)
            ax.text(x_pos2, zy_values[0] + 0.01, f'{curDim}', ha="center", va="bottom", rotation=90)
    
    ax.set_xticks([i + 3.5 * width for i in range(len(compRange))])
    ax.set_xticklabels(compRange)
    
        # Create custom color patches for legends
    legend_patches = [
        Patch(color=cmap(0.35), label=modelNames[0]),
        Patch(color=cmap2(0.35), label=modelNames[1])
    ]

    plt.legend(handles=legend_patches)  # Use custom color patches for legends
    
    
    plt.grid(True)
    plt.xlabel('Number of components')
    plt.ylabel('minDCF')
    plt.title(title)
    
    max_height = max(max(zip(data[:, 2], zNormData[:, 2])))
    plt.ylim(0, max_height * 1.15)  # Adjust the multiplier (1.2) as needed
    
    plt.savefig(f'results/img/{title}.png')
    plt.show()
    plt.clf()