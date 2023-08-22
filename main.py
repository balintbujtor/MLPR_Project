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

    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    #kfold
    nFolds = 5
    
    znorm = False
    
    saveResults = True
    showResults = False
    
    runInitAnalysis = False
    runGenerative = False
    runLogReg = True
    runSVM = False

    if runInitAnalysis:
        runInitAnalysis(DTR, LTR)
    
    if runSVM:
        
        minDCFarrayLinSVM = []
        minDCFarrayPoliSVM = [] 
        minDCFarrayRbfSVM = []
        
        # Linear SVM
        for dim in range(11, 7, -1):
            
            # no pca
            if(dim == 11):
                if znorm:
                    DTR, _, _ = preproc.zNormalization(DTR)
                kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
            else:
                if znorm:
                    DTR, _, _ = preproc.zNormalization(DTR)

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
        
        if showResults:
            
            LinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [LinSVMResults, zLinSVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["Linear SVM", "Z-normed LinSVM"], f'Linear SVM minDCFs - effPrior: {formattedPrior}', "C", range(11, 7, -1))

        print('trained')
    
    
    if runGenerative:
        minDCFMVGarray, minDCFTiedGarray, minDCFNBarray, minDCFTiedNBArray = generativeModels.trainAllGenerativeClassifiers(11, 5, DTR, LTR, [prior, Cfn, Cfp], nFolds, znorm=False)
        
        if saveResults:
            helpers.saveResults("MVG", minDCFMVGarray, effPrior, znorm)
            helpers.saveResults("TiedG", minDCFTiedGarray, effPrior, znorm)
            helpers.saveResults("NB", minDCFNBarray, effPrior, znorm)
            helpers.saveResults("TiedNB", minDCFTiedNBArray, effPrior, znorm)

        print("finished")
    
    
    
    if runLogReg:
        minDCFarrayLogReg = logReg.trainLogRegClassifiers(11, 7, DTR, LTR, [prior, Cfn, Cfp], nFolds, znorm, quadratic=False)
        minDCFarrayQLogReg = logReg.trainLogRegClassifiers(11, 7, DTR, LTR, [prior, Cfn, Cfp], nFolds, znorm, quadratic=True)
        
        if saveResults:
            formattedPrior = "{:.2f}".format(effPrior)

            helpers.saveMinDCF("LogReg", minDCFarrayLogReg, formattedPrior, znorm)
            helpers.saveMinDCF("QLogReg", minDCFarrayQLogReg, formattedPrior, znorm)
            
        if showResults:
            logRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLogRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [logRegResults, zLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["LR", "z-LR"], f'Linear Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

            qLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zQLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [qLogRegResults, zQLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["Q-LR", "z-Q-LR"], f'Quadratic Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

        print('trained')
        
        pTs = [0.05, 0.5, 0.9]

        minDCFarrayLogReg8 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-5, znorm=True, quadratic=False, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogReg9 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=9, l=10e-2, znorm=False, quadratic=False, pTs=pTs, nFolds=nFolds)
        
        minDCFarrayLogRegQ8 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-2, znorm=False, quadratic=True, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogRegQ10 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=11, l=10e-3, znorm=True, quadratic=True, pTs=pTs, nFolds=nFolds)
        
        if saveResults:
            
            helpers.saveMinDCF("LogRegPCA8_lambda10e-5", minDCFarrayLogReg8, prior, zNorm=True)
            helpers.saveMinDCF("LogRegPCA9_lambda10e-2", minDCFarrayLogReg9, prior, zNorm=False)
            
            helpers.saveMinDCF("QLogRegPCA8_lambda10e-2", minDCFarrayLogRegQ8, prior, zNorm=False)
            helpers.saveMinDCF("QLogRegPCA10_lambda10e-3", minDCFarrayLogRegQ10, prior, zNorm=True)
