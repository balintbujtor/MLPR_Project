import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import generativeModels.generativeModels as generativeModels
import logisticRegression.logisticRegression as logReg
import matplotlib.pyplot as plot
import visualization.visualization as vis
import evaluation.evaluation as eval


visualize = False

if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')
    DTE, LTE = loader.loadData('data/Test.txt')

    # initial preprocessing
    
    PCAdirs, _ = preproc.computePCA(DTR, 2)
    
    pcaClass0 = PCAdirs[:, LTR == 0]
    pcaClass1 = PCAdirs[:, LTR == 1]
    pcaClass0attr0 = pcaClass0[0, :]
    pcaClass1attr0 = pcaClass1[0, :]
    pcaClass0attr1 = pcaClass0[1, :]
    pcaClass1attr1 = pcaClass1[1, :]
    cumRatios = preproc.computeCumVarRatios(DTR)
    
    LDAdir = preproc.computeLDA(DTR, LTR, 1)
    ldaClass0 = LDAdir[:, LTR == 0]
    ladClass1 = LDAdir[:, LTR == 1]
    ldaClass0attr0 = ldaClass0[0, :]
    ldaClass1attr0 = ladClass1[0, :] 
    
    if visualize:
        vis.plotHistogram(pcaClass0attr0, pcaClass1attr0)
        vis.plotHistogram(pcaClass0attr1, pcaClass1attr1)
        vis.plotScatter(pcaClass0, pcaClass1)
        vis.plotHistogram(ldaClass0attr0, ldaClass1attr0)
        vis.plotCorrMat(DTR[:, LTR == 0])
        vis.plotCorrMat(DTR[:, LTR == 1])
        vis.plotCorrMat(DTR)
        vis.plotCumVarRatios(cumRatios, DTR.shape[0] + 1)


    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    
    #kfold

    nFolds = 5
    
    runGenerative = False
    minDCFMVGarray = []
    minDCFTiedMVGarray = []
    minDCFNaiveMVGarray = []

    # to try out different pca directions
    for j in range(10, 5, -1):
        
        if(j == 10):
            # without PCA
            kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
        
        else:
            reducedData, _ = preproc.computePCA(DTR, j)
            kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)

        sLogPostMVG = []
        sLogPostTiedMVG = []
        sLogPostNaiveMVG = []
        
        correctEvalLabels = []

        for i in range(0, nFolds):
            
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
            
            # same for all, do it only once
            correctEvalLabels.append(evalLabels)
            
            if runGenerative:
                # MVG with K-fold
                _, sPostLogMVG = generativeModels.logMVG(trainingData, trainingLabels, evalData, 2, priorProb)
                sLogPostMVG.append(sPostLogMVG)
                
                # tied MVG
                _, sPostLogTiedMVG = generativeModels.logTiedMVG(trainingData, trainingLabels, evalData, 2, priorProb)
                sLogPostTiedMVG.append(sPostLogTiedMVG)
                
                # naive
                _, sPostLogNaiveMVG = generativeModels.logNaiveBayes(trainingData, trainingLabels, evalData, 2, priorProb)
                sLogPostNaiveMVG.append(sPostLogNaiveMVG)
                
                #naivetied - we are not going to do it because the tied model performs worse than the untied
            

            #logreg
            
            #logreg znorm
        
        correctEvalLabels = np.hstack(correctEvalLabels)

        if runGenerative:
            # eval of MVG
            sLogPostMVG = np.hstack(sLogPostMVG)
            llrMVG = np.log(sLogPostMVG[1] / sLogPostMVG[0])
            minDCFMVG = eval.computeMinDCF(llrMVG, correctEvalLabels, prior, Cfn, Cfp)
            minDCFMVGarray.append([j, minDCFMVG])
            
            # eval of tied MVG
            sLogPostTiedMVG = np.hstack(sLogPostTiedMVG)
            llrTiedMVG = np.log(sLogPostTiedMVG[1] / sLogPostTiedMVG[0])
            minDCFTiedMVG = eval.computeMinDCF(llrTiedMVG, correctEvalLabels, prior, Cfn, Cfp)
            minDCFTiedMVGarray.append([j, minDCFTiedMVG])
            
            # eval of tied MVG
            sLogPostNaiveMVG = np.hstack(sLogPostNaiveMVG)
            llrNaiveMVG = np.log(sLogPostNaiveMVG[1] / sLogPostNaiveMVG[0])
            minDCFNaiveMVG = eval.computeMinDCF(llrNaiveMVG, correctEvalLabels, prior, Cfn, Cfp)
            minDCFNaiveMVGarray.append([j, minDCFNaiveMVG])
        
        


    print("finished")