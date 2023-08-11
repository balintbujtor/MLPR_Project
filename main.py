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
    
    saveResults = True
    
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
        
        if saveResults:
            
            np.save(f"results/minDCFMVGarray_Znorm{znorm}_prior{effPrior}", minDCFarrayLinSVM)
            np.savetxt(f"results/minDCFMVGarray_Znorm{znorm}_prior{effPrior}", minDCFarrayLinSVM)
            
        print('trained')
    
    
    # logistic regression
    runLogReg = True
    znorm = False
    if runLogReg:
        
        minDCFarrayLogReg = []
        minDCFarrayQLogReg = []
        
        for j in range(11, 7, -1):
            
            # no pca
            if(j == 11):
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            else:
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)

                reducedData, _ = preproc.computePCA(DTR, j)
                kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
                
            llrsLogReg = []
            llrsQLogReg = []
            
            lambdas = np.logspace(-6, 4, 11)
            
            for l in range(len(lambdas)):
                curLambda = lambdas[l]
                
                correctEvalLabels = []
                
                llrsLogReg.append([curLambda, []])
                llrsQLogReg.append([curLambda, []])
                
                for i in range(0, nFolds):
                    
                    trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                    correctEvalLabels.append(evalLabels)

                    # training Linear Logistic Regression
                    logRegObj = logReg.logRegClassifier(DTR=trainingData, LTR=trainingLabels, l=curLambda, pi=effPrior)
                    wtrain, btrain = logRegObj.trainBin()  
                    logScores, preds = logRegObj.evaluateBin(DTE=evalData, w=wtrain, b=btrain, thr=np.log(effPrior))
                    
                    llrsLogReg[l][1].append(logScores)
                    
                    # training Quadratic Logistic Regression
                    qtrainingData, qEvalData = logReg.transformTrainAndTestToQuadratic(trainingData, evalData)
                    
                    qLogRegObj = logReg.logRegClassifier(DTR=qtrainingData, LTR=trainingLabels, l=curLambda, pi=effPrior)
                    qwtrain, qbtrain = qLogRegObj.trainBin()
                    qlogScores, qpreds = qLogRegObj.evaluateBin(DTE=qEvalData, w=qwtrain, b=qbtrain, thr=np.log(effPrior))
                    
                    llrsQLogReg[l][1].append(qlogScores)

                
            correctEvalLabels = np.hstack(correctEvalLabels)
            for i in range(len(llrsLogReg)):
                
                llrsLogReg[i][1] = np.hstack(llrsLogReg[i][1])
                minDCFLogReg = eval.computeMinDCF(llrsLogReg[i][1], correctEvalLabels, prior, Cfn, Cfp)
                minDCFarrayLogReg.append([j, llrsLogReg[i][0], minDCFLogReg])
                
                llrsQLogReg[i][1] = np.hstack(llrsQLogReg[i][1])
                minDCFQLogReg = eval.computeMinDCF(llrsQLogReg[i][1], correctEvalLabels, prior, Cfn, Cfp)
                minDCFarrayQLogReg.append([j, llrsQLogReg[i][0], minDCFQLogReg])
        
        if saveResults:
            formattedPrior = "{:.2f}".format(effPrior)

            np.save(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{znorm}", minDCFarrayLogReg)
            np.savetxt(f"results/txt/minDCFLogReg_prior{formattedPrior}_Znorm{znorm}", minDCFarrayLogReg)
            
            np.save(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{znorm}", minDCFarrayQLogReg)
            np.savetxt(f"results/txt/minDCFQLogReg_prior{formattedPrior}_Znorm{znorm}", minDCFarrayQLogReg)
            
        print('trained')

    showResults = True
    if showResults:
        formattedPrior = "{:.2f}".format(effPrior)
        znorm = True
        logRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{znorm}.npy")
        vis.plotLogRegDCFs(logRegResults, "Logistic Regression", "lambdas", range(11, 7, -1))
    
    runGenerative = False
    if runGenerative:
        
        znorm = True
        minDCFMVGarray = []
        minDCFTiedGarray = []
        minDCFNBarray = []
        minDCFTiedNBArray = []

        # to try out different pca directions
        for j in range(11, 5, -1):
            
            if(j == 11):
                # without PCA
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            
            else:
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
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

        if saveResults:
            formattedPrior = "{:.2f}".format(effPrior)
            
            np.save(f"results/npy/minDCFMVG_prior{formattedPrior}_Znorm{znorm}", minDCFMVGarray)
            np.savetxt(f"results/txt/minDCFMVG_prior{formattedPrior}_Znorm{znorm}", minDCFMVGarray)

            np.save(f"results/npy/minDCFTiedG_prior{formattedPrior}_Znorm{znorm}", minDCFTiedGarray)
            np.savetxt(f"results/txt/minDCFTiedG_prior{formattedPrior}_Znorm{znorm}", minDCFTiedGarray)
            
            np.save(f"results/npy/minDCFNB_prior{formattedPrior}_Znorm{znorm}", minDCFNBarray)
            np.savetxt(f"results/txt/minDCFNB_prior{formattedPrior}_Znorm{znorm}", minDCFNBarray)
            
            np.save(f"results/npy/minDCFTiedNB_prior{formattedPrior}_Znorm{znorm}", minDCFTiedNBArray)
            np.savetxt(f"results/txt/minDCFTiedNB_prior{formattedPrior}_Znorm{znorm}", minDCFTiedNBArray)


        print("finished")