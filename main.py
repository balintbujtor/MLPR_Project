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
    plotResults = False
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
    runSVM = False
    znorm = False
    if runSVM:
        
        minDCFarrayLinSVM = []
        minDCFarrayPoliSVM = [] 
        minDCFarrayRbfSVM = []
        
        # Linear SVM
        for dim in range(11, 7, -1):
            
            # no pca
            if(dim == 11):
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            else:
                if znorm:
                    DTR, _, _ = helpers.ZNormalization(DTR)

                reducedData, _ = preproc.computePCA(DTR, dim)
                kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
                
            llrsLinSVM = []
            
            Cs = np.logspace(-6, 4, 11)
            
            for c in range(len(Cs)):
                
                curC = Cs[c]
                correctEvalLabels = []
                llrsLinSVM.append([curC, []])
                
                for i in range(0, nFolds):
                    
                    trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                    correctEvalLabels.append(evalLabels)

                    # training Linear SVM, without class rebalancing
                    linSVMObj = svm.LinearSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                    linAlpha, linW, linPrimal, linDual = linSVMObj.train(curC)
                    linLogScores, linPreds = linSVMObj.predict(evalData, linW)
                    
                    llrsLinSVM[c][1].append(linLogScores)
                    
                    # TODO: move it to different kfold training loop
                    # training Polynomial SVM without class rebalancing
                    # poliSVMObj = svm.KernelSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                    # poliKernel = svm.polyKernelWrapper(1,2,0)
                    # poliAlpha, poliW, poliPrimal, poliDual = poliSVMObj.train(curC, kernel=poliKernel)
                    # poliLogScores, poliPreds = poliSVMObj.predict(evalData, poliW, kernel=poliKernel)

                
            correctEvalLabels = np.hstack(correctEvalLabels)
            for i in range(len(llrsLinSVM)):
                
                llrsLinSVM[i][1] = np.hstack(llrsLinSVM[i][1])
                minDCFLinSVM = eval.computeMinDCF(llrsLinSVM[i][1], correctEvalLabels, prior, Cfn, Cfp)
                minDCFarrayLinSVM.append([dim, llrsLinSVM[i][0], minDCFLinSVM])
        
        
        formattedPrior = "{:.2f}".format(effPrior)
        if saveResults:

            np.save(f"results/npy/minDCFLinSVM_prior{formattedPrior}_Znorm{znorm}", minDCFarrayLinSVM)
            np.savetxt(f"results/txt/minDCFLinSVM_prior{formattedPrior}_Znorm{znorm}", minDCFarrayLinSVM)
        
        if plotResults:
            
            LinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [LinSVMResults, zLinSVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["Linear SVM", "Z-normed LinSVM"], f'Linear SVM minDCFs - effPrior: {formattedPrior}', "C", range(11, 7, -1))

        print('trained')
    
    
    # logistic regression
    runLogReg = False
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
                    logScores, _ = logRegObj.evaluateBin(DTE=evalData, w=wtrain, b=btrain)
                    
                    llrsLogReg[l][1].append(logScores)
                    
                    # training Quadratic Logistic Regression
                    qtrainingData, qEvalData = logReg.transformTrainAndTestToQuadratic(trainingData, evalData)
                    
                    qLogRegObj = logReg.logRegClassifier(DTR=qtrainingData, LTR=trainingLabels, l=curLambda, pi=effPrior)
                    qwtrain, qbtrain = qLogRegObj.trainBin()
                    qlogScores, _ = qLogRegObj.evaluateBin(DTE=qEvalData, w=qwtrain, b=qbtrain)
                    
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

            logRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLogRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [logRegResults, zLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["LR", "z-LR"], f'Linear Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

            qLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zQLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [qLogRegResults, zQLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["Q-LR", "z-Q-LR"], f'Quadratic Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

        print('trained')

    trainBestLogRegWDifferentPT = False
    if trainBestLogRegWDifferentPT:

        # model A
        reducedData9, _ = preproc.computePCA(DTR, 9)
        kdata9, klabels9 = helpers.splitToKFold(reducedData9, LTR, K=nFolds)

        # model B
        DTRZ, _, _ = helpers.ZNormalization(DTR)
        reducedData8, _ = preproc.computePCA(DTRZ, 8)
        kdata8, klabels8 = helpers.splitToKFold(reducedData8, LTR, K=nFolds)
        
        pTs = [0.05, 0.5, 0.9]

        minDCFarrayLogReg8 = []
        minDCFarrayLogReg9 = []
        llrsLogReg8 = []
        llrsLogReg9 = []
        
        for pT in pTs:
            
            llrsLogReg8.append([pT, []])
            llrsLogReg9.append([pT, []])
            correctEvalLabels9 = []
            correctEvalLabels8 = []
            
            for i in range(0, nFolds):
                
                trainingData9, trainingLabels9, evalData9, evalLabels9 = helpers.getCurrentKFoldSplit(kdata9, klabels9, i, nFolds)
                trainingData8, trainingLabels8, evalData8, evalLabels8 = helpers.getCurrentKFoldSplit(kdata8, klabels8, i, nFolds)
                correctEvalLabels9.append(evalLabels9)
                correctEvalLabels8.append(evalLabels8)
                
                # training Linear Logistic Regression candidate A
                logRegObj9 = logReg.logRegClassifier(DTR=trainingData9, LTR=trainingLabels9, l=10e-2, pi=pT)
                wtrain9, btrain9 = logRegObj9.trainBin()
                logScores9, _ = logRegObj9.evaluateBin(DTE=evalData9, w=wtrain9, b=btrain9)
                llrsLogReg9[-1][1].append(logScores9)
                
                # training Linear Logistic Regression candidate B
                logRegObj8 = logReg.logRegClassifier(DTR=trainingData8, LTR=trainingLabels8, l=10e-5, pi=pT)
                wtrain8, btrain8 = logRegObj8.trainBin()
                logScores8, _ = logRegObj8.evaluateBin(DTE=evalData8, w=wtrain8, b=btrain8)           
                llrsLogReg8[-1][1].append(logScores8)
                
        
        correctEvalLabels9 = np.hstack(correctEvalLabels9)
        correctEvalLabels8 = np.hstack(correctEvalLabels8) 
        for i in range(len(llrsLogReg8)):
            
            llrsLogReg8[i][1] = np.hstack(llrsLogReg8[i][1])
            minDCFLogReg8 = eval.computeMinDCF(llrsLogReg8[i][1], correctEvalLabels8, prior, Cfn, Cfp)
            minDCFarrayLogReg8.append([llrsLogReg8[i][0], minDCFLogReg8])
            
            llrsLogReg9[i][1] = np.hstack(llrsLogReg9[i][1])
            minDCFLogReg9 = eval.computeMinDCF(llrsLogReg9[i][1], correctEvalLabels9, prior, Cfn, Cfp)
            minDCFarrayLogReg9.append([llrsLogReg9[i][0], minDCFLogReg9])
    
        if saveResults:
            
            formattedPrior = "{:.2f}".format(effPrior)
            np.save(f"results/npy/minDCFLogRegPCA8_lambda10e-5_prior{formattedPrior}_Znorm{True}", minDCFarrayLogReg8)
            np.savetxt(f"results/txt/minDCFLogRegPCA8_lambda10e-5_prior{formattedPrior}_Znorm{True}", minDCFarrayLogReg8)
            
            np.save(f"results/npy/minDCFLogRegPCA9_lambda10e-2_prior{formattedPrior}_Znorm{False}", minDCFarrayLogReg9)
            np.savetxt(f"results/txt/minDCFLogRegPCA9_lambda10e-2_prior{formattedPrior}_Znorm{False}", minDCFarrayLogReg9)

    # TODO: refactor this
    trainBestQLogRegWDifferentPT = True
    if trainBestQLogRegWDifferentPT:
        # model C
        reducedDataQ8, _ = preproc.computePCA(DTR, 8)
        kdataQ8, klabelsQ8 = helpers.splitToKFold(reducedDataQ8, LTR, K=nFolds)

        # model D
        DTRZ, _, _ = helpers.ZNormalization(DTR)
        kdataQ10, klabelsQ10 = helpers.splitToKFold(DTRZ, LTR, K=nFolds)
        
        pTs = [0.05, 0.5, 0.9]

        minDCFarrayLogRegQ8 = []
        minDCFarrayLogRegQ10 = []
        llrsLogRegQ8 = []
        llrsLogRegQ10 = []
        
        for pT in pTs:
            
            llrsLogRegQ8.append([pT, []])
            llrsLogRegQ10.append([pT, []])
            correctEvalLabelsQ8 = []
            correctEvalLabelsQ10 = []
            
            for i in range(0, nFolds):
                
                trainingDataQ8, trainingLabelsQ8, evalDataQ8, evalLabelsQ8 = helpers.getCurrentKFoldSplit(kdataQ8, klabelsQ8, i, nFolds)
                trainingDataQ10, trainingLabelsQ10, evalDataQ10, evalLabelsQ10 = helpers.getCurrentKFoldSplit(kdataQ10, klabelsQ10, i, nFolds)
                correctEvalLabelsQ8.append(evalLabelsQ8)
                correctEvalLabelsQ10.append(evalLabelsQ10)
                
                # training QUADRATIC Logistic Regression candidate C
                qtrainingDataQ8, qEvalDataQ8 = logReg.transformTrainAndTestToQuadratic(trainingDataQ8, evalDataQ8)
                logRegObjQ8 = logReg.logRegClassifier(DTR=qtrainingDataQ8, LTR=trainingLabelsQ8, l=10e-2, pi=pT)
                wtrainQ8, btrainQ8 = logRegObjQ8.trainBin()
                logScoresQ8, _ = logRegObjQ8.evaluateBin(DTE=qEvalDataQ8, w=wtrainQ8, b=btrainQ8)
                llrsLogRegQ8[-1][1].append(logScoresQ8)
                
                # training QUDARATIC Logistic Regression candidate D
                qtrainingDataQ10, qEvalDataQ10 = logReg.transformTrainAndTestToQuadratic(trainingDataQ10, evalDataQ10)
                logRegObjQ10 = logReg.logRegClassifier(DTR=qtrainingDataQ10, LTR=trainingLabelsQ10, l=10e-3, pi=pT)
                wtrainQ10, btrainQ10 = logRegObjQ10.trainBin()
                logScoresQ10, _ = logRegObjQ10.evaluateBin(DTE=qEvalDataQ10, w=wtrainQ10, b=btrainQ10)           
                llrsLogRegQ10[-1][1].append(logScoresQ10)
                
        
        correctEvalLabelsQ8 = np.hstack(correctEvalLabelsQ8)
        correctEvalLabelsQ10 = np.hstack(correctEvalLabelsQ10) 
        for i in range(len(llrsLogRegQ8)):
            
            llrsLogRegQ8[i][1] = np.hstack(llrsLogRegQ8[i][1])
            minDCFLogRegQ8 = eval.computeMinDCF(llrsLogRegQ8[i][1], correctEvalLabelsQ8, prior, Cfn, Cfp)
            minDCFarrayLogRegQ8.append([llrsLogRegQ8[i][0], minDCFLogRegQ8])
            
            llrsLogRegQ10[i][1] = np.hstack(llrsLogRegQ10[i][1])
            minDCFLogRegQ10 = eval.computeMinDCF(llrsLogRegQ10[i][1], correctEvalLabelsQ10, prior, Cfn, Cfp)
            minDCFarrayLogRegQ10.append([llrsLogRegQ10[i][0], minDCFLogRegQ10])
    
        if saveResults:
            
            formattedPrior = "{:.2f}".format(effPrior)
            np.save(f"results/npy/minDCFQuadLogRegPCA8_lambda10e-2_prior{formattedPrior}_Znorm{False}", minDCFarrayLogRegQ8)
            np.savetxt(f"results/txt/minDCFQuadLogRegPCA8_lambda10e-2_prior{formattedPrior}_Znorm{False}", minDCFarrayLogRegQ8)
            
            np.save(f"results/npy/minDCFQuadLogRegNoPCA_lambda10e-3_prior{formattedPrior}_Znorm{True}", minDCFarrayLogRegQ10)
            np.savetxt(f"results/txt/minDCFQuadLogRegNoPCA_lambda10e-3_prior{formattedPrior}_Znorm{True}", minDCFarrayLogRegQ10)
            
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