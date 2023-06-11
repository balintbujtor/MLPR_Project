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
    postlogscores = []
    correctEvalLabels = []
    nFolds = 5

    kdata, klabels = helpers.splitToKFold(DTR, LTR)

    for i in range(0, nFolds):
        
        trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
        
        # MVG with K-fold
        _, sPostLogMVG = generativeModels.logMVG(trainingData, trainingLabels, evalData, 2, priorProb)
        postlogscores.append(sPostLogMVG)
        correctEvalLabels.append(evalLabels)

    postlogscores = np.hstack(postlogscores)
    llr = np.log(postlogscores[1] / postlogscores[0])
    
    predLabels = np.argmax(postlogscores, 0)
    correctEvalLabels = np.hstack(correctEvalLabels)
    
    acc = eval.compute_accuracy(predLabels, correctEvalLabels)
    minDCFMVg = eval.computeMinDCF(llr, correctEvalLabels, prior, Cfn, Cfp)

    print("finished")