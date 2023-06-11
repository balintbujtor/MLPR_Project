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
    
    PCAdirs = preproc.computePCA(DTR, 2)
    pcaClass0 = PCAdirs[:, LTR == 0]
    pcaClass1 = PCAdirs[:, LTR == 1]
    pcaClass0attr0 = pcaClass0[0, :]
    pcaClass1attr0 = pcaClass1[0, :]
    pcaClass0attr1 = pcaClass0[1, :]
    pcaClass1attr1 = pcaClass1[1, :]
    
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


    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    
    #kfold
    # TODO: check k fold split, how it should behave correctly
    postlogscores = []
    correctEvalLabels = []
    nFolds = 5
    nsplits, splitbounds = helpers.splitToKFold(DTR.shape[1], nFolds)
    kdata, klabels = helpers.KsplitNew(DTR, LTR, 0, 5)

    for i in range(0, nFolds):
        
        trainingData = []
        trainingLabels = []
        evalData = []
        evalLabels = []
        
        for j in range(nFolds):
            if j != i:
                trainingData.append(kdata[j])
                trainingLabels.append(klabels[j])
            else:
                evalData = kdata[i]
                evalLabels = klabels[i]
        
        trainingData = np.hstack(trainingData)
        trainingLabels = np.hstack(trainingLabels)
        
        # MVG with K-fold
        _, sPostLogMVG = generativeModels.logMVG(trainingData, trainingLabels, evalData, 2, priorProb)
        postlogscores.append(sPostLogMVG)
        correctEvalLabels.append(evalLabels)

    postlogscores = np.hstack(postlogscores)
    llr = np.log(postlogscores[0] / postlogscores[1])
    
    correctEvalLabels = np.hstack(correctEvalLabels)
    minDCFMVg = eval.computeMinDCF(correctEvalLabels, correctEvalLabels, prior, Cfn, Cfp)

    print("finished")