import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import generativeModels.generativeModels as generativeModels
import logisticRegression.logisticRegression as logReg
import supportVectorMachines.svm as svm
import visualization.visualization as vis
import evaluation.evaluation as eval



if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')
    # DTE, LTE = loader.loadData('data/Test.txt')

    runInitAnalysis = False
    if runInitAnalysis:
        # initial preprocessing
        class0 = DTR[:, LTR == 0]
        class1 = DTR[:, LTR == 1]
        for i in range(DTR.shape[0]):
            class0attri = class0[i, :]
            class1attri = class1[i, :]
            vis.plotHistogram(class0attri, class1attri, f"HistogramDatasetFeature_{i}")
            
        PCAdirs, _ = preproc.computePCA(DTR, 2)
        
        pcaClass0 = PCAdirs[:, LTR == 0]
        pcaClass1 = PCAdirs[:, LTR == 1]
        for i in range(2):
            pcaClass0attri = pcaClass0[i, :]
            pcaClass1attri = pcaClass1[i, :]
            vis.plotHistogram(pcaClass0attri, pcaClass1attri, f"HistogramPCAdir{i}")
            
        vis.plotScatter(pcaClass0, pcaClass1, "scatterPCAdirs")

        LDAdir = preproc.computeLDA(DTR, LTR, 1)
        ldaClass0 = LDAdir[:, LTR == 0]
        ladClass1 = LDAdir[:, LTR == 1]
        ldaClass0attr0 = ldaClass0[0, :]
        ldaClass1attr0 = ladClass1[0, :] 
        vis.plotHistogram(ldaClass0attr0, ldaClass1attr0, "LDA_direction")

        cumRatios = preproc.computeCumVarRatios(DTR)
        vis.plotCumVarRatios(cumRatios, DTR.shape[0] + 1)
        
        vis.plotCorrMat(DTR[:, LTR == 0], "PearsonCorrelationClass0")
        vis.plotCorrMat(DTR[:, LTR == 1], "PearsonCorrelationClass1")
        vis.plotCorrMat(DTR, "PearsonCorrelation")


    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    
    #kfold

    nFolds = 5
    
    #SVM
    
    #linear
        #PCA 9, 7 - no pca
        #znorm -no znorm
    runSVM = False
    if runSVM:
        
        pcaTrials = [10, 9, 7]
        znorm = False
        minDCFarrayLinSVM = []
        
        for pcadim in pcaTrials:

            if(pcadim == 10):
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            else:
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)

                reducedData, _ = preproc.computePCA(DTR, pcadim)
                kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
            
            
            correctEvalLabels = []
            llrsLinearSVM = []
            
            for i in range(0, nFolds):
            
                trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                
                for c_iter in range(0, 6):
                    
                    curC = 10**c_iter*1e-4
                    if(i == 0):
                        llrsLinearSVM.append([curC, []])
                    
                    linSVMObj = svm.LinearSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                    alphaStar, wStar, primalLoss, dualLoss = linSVMObj.train(curC)
                    logScores, preds = linSVMObj.predict(evalData, wStar, np.log(effPrior))
                    
                    llrsLinearSVM[c_iter][1].append(logScores)
                
                correctEvalLabels.append(evalLabels)
                
            correctEvalLabels = np.hstack(correctEvalLabels)
            for i in range(len(llrsLinearSVM)):
                llrsLinearSVM[i][1] = np.hstack(llrsLinearSVM[i][1])
                
                minDCFLinSVM = eval.computeMinDCF(llrsLinearSVM[i][1], correctEvalLabels, prior, Cfn, Cfp)
                minDCFarrayLinSVM.append([znorm, pcadim, llrsLinearSVM[i][0], minDCFLinSVM])
        
        print('trained')
    
    
    # logistic regression
    runLogReg = False
    znorm = False
    if runLogReg:
        
        minDCFarrayLogReg = []
        minDCFarrayZLogReg = []
        
        for j in range(10, 6, -1):
            
            # no pca
            if(j == 10):
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            else:
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)

                reducedData, _ = preproc.computePCA(DTR, j)
                kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
                
            correctEvalLabels = []
            llrsLogReg = []
            
            for i in range(0, nFolds):
                
                trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                
                for l_iter in range(0, 6):
                    
                    curLambda = 10**l_iter*1e-5
                    if(i == 0):
                        llrsLogReg.append([curLambda, []])
                    
                    # TODO: check with Kasia if prior stuff is ok
                    logRegObj = logReg.logRegClassifier(DTR=trainingData, LTR=trainingLabels, l=curLambda, pi=effPrior)
                    wtrain, btrain = logRegObj.trainBin()  
                    logScores, preds = logRegObj.evaluateBin(DTE=evalData, w=wtrain, b=btrain, thr=np.log(effPrior))
                    
                    llrsLogReg[l_iter][1].append(logScores)
                
                correctEvalLabels.append(evalLabels)

                
            correctEvalLabels = np.hstack(correctEvalLabels)
            for i in range(len(llrsLogReg)):
                llrsLogReg[i][1] = np.hstack(llrsLogReg[i][1])
                
                minDCFLogReg = eval.computeMinDCF(llrsLogReg[i][1], correctEvalLabels, prior, Cfn, Cfp)
                minDCFarrayLogReg.append([znorm, j, llrsLogReg[i][0], minDCFLogReg])
        
        print('trained')

    
    
    
    runGenerative = True
    if runGenerative:
        
        znorm = False
        minDCFMVGarray = []
        minDCFTiedMVGarray = []
        minDCFNaiveMVGarray = []

        # to try out different pca directions
        for j in range(10, 5, -1):
            
            if(j == 10):
                # without PCA
                if znorm:
                    DTR = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            
            else:
                if znorm:
                    DTR = helpers.ZNormalization(DTR)
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
            
            correctEvalLabels = np.hstack(correctEvalLabels)

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

        save = True
        if save:
            np.save(f"results/minDCFMVGarray_Znorm{znorm}_prior{effPrior}", minDCFMVGarray)
            np.savetxt(f"results/minDCFMVGarray_Znorm{znorm}_prior{effPrior}", minDCFMVGarray)

            np.save(f"results/minDCFTiedMVGarray_Znorm{znorm}_prior{effPrior}", minDCFTiedMVGarray)
            np.savetxt(f"results/minDCFTiedMVGarray_Znorm{znorm}_prior{effPrior}", minDCFTiedMVGarray)
            
            np.save(f"results/minDCFNaiveMVGarray_Znorm{znorm}_prior{effPrior}", minDCFNaiveMVGarray)
            np.savetxt(f"results/minDCFNaiveMVGarray_Znorm{znorm}_prior{effPrior}", minDCFNaiveMVGarray)
            


        print("finished")