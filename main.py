import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import generativeModels.generativeModels as generativeModels
import logisticRegression.logisticRegression as logReg
import supportVectorMachines.svm as svm
import visualization.visualization as vis
import gaussianMixtureModels.gmm as gmm
import calibrationFusion.calibrationFusion as calFuse
import testing.testing as testing

if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')

    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    formattedPrior = "{:.2f}".format(effPrior)

    #kfold
    nFolds = 5
    
    znorm = False
    
    saveResults = True
    showResults = False
    
    runInitAnalysis = False
    runGenerative = False
    runLogReg = False
    runSVM = False
    runGMM = False
    runCalibration = False
    runEvaluation = True
    
    if runInitAnalysis:
        runInitAnalysis(DTR, LTR)
    
    if runGenerative:
        minDCFMVGarray, minDCFTiedGarray, minDCFNBarray, minDCFTiedNBArray = generativeModels.trainAllGenerativeClassifiers(
            11, 5, DTR, LTR, [prior, Cfn, Cfp], nFolds, znorm=False
        )
        
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

            helpers.saveMinDCF("LogReg", minDCFarrayLogReg, effPrior, znorm)
            helpers.saveMinDCF("QLogReg", minDCFarrayQLogReg, effPrior, znorm)
            
        if showResults:

            logRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLogRegResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [logRegResults, zLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["LR", "z-LR"], f'Linear Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

            qLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zQLogRegResults = np.load(f"results/npy/minDCFQLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [qLogRegResults, zQLogRegResults]
            vis.plotLogRegDCFs(modelsToShow, ["Q-LR", "z-Q-LR"], f'Quadratic Logistic Regression minDCFs - effPrior: {formattedPrior}', "lambda", range(11, 7, -1))

    if runLogReg:
        
        print('trained')
        
        pTs = [0.05, 0.5, 0.9]

        minDCFarrayLogReg8 = logReg.trainSingleLogRegClassifier(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-5, znorm=True, quadratic=False, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogReg9 = logReg.trainSingleLogRegClassifier(DTR, LTR, [prior, Cfn, Cfp], PCADir=9, l=10e-2, znorm=False, quadratic=False, pTs=pTs, nFolds=nFolds)
        
        minDCFarrayLogRegQ8 = logReg.trainSingleLogRegClassifier(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-2, znorm=False, quadratic=True, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogRegQ10 = logReg.trainSingleLogRegClassifier(DTR, LTR, [prior, Cfn, Cfp], PCADir=11, l=10e-3, znorm=True, quadratic=True, pTs=pTs, nFolds=nFolds)
        
        if saveResults:
            
            helpers.saveMinDCF("LogRegPCA8_lambda10e-5", minDCFarrayLogReg8, effPrior, zNorm=True)
            helpers.saveMinDCF("LogRegPCA9_lambda10e-2", minDCFarrayLogReg9, effPrior, zNorm=False)
            
            helpers.saveMinDCF("QLogRegPCA8_lambda10e-2", minDCFarrayLogRegQ8, effPrior, zNorm=False)
            helpers.saveMinDCF("QLogRegPCA10_lambda10e-3", minDCFarrayLogRegQ10, effPrior, zNorm=True)

    if runSVM:

        minDCFarrayLinSVM1 = svm.trainLinSVMClassifiers(11, 7, DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=False, Csparam=[-5, 1, 7])
        minDCFarrayLinSVM2 = svm.trainLinSVMClassifiers(11, 7, DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=True, Csparam=[-5, 1, 7])
        minDCFarrayLinSVM3 = svm.trainLinSVMClassifiers(11, 7, DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=False, Csparam=[-5, 1, 7])
        minDCFarrayLinSVM4 = svm.trainLinSVMClassifiers(11, 7, DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=True, Csparam=[-5, 1, 7])

        # quadratic polynomial kernel
        d = 2
        polyKernel = svm.polyKernelWrapper(1, d, 0)
        minDCFarrayPolySVM1 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=False, Csparam=[-5, 1, 7], kernel=polyKernel)
        minDCFarrayPolySVM2 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=True, Csparam=[-5, 1, 7], kernel=polyKernel)
        minDCFarrayPolySVM3 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=False, Csparam=[-5, 1, 7], kernel=polyKernel)
        minDCFarrayPolySVM4 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=True, Csparam=[-5, 1, 7], kernel=polyKernel)
        
        # rbf kernel
        gammaArray = [0.00001, 0.001, 0.1]
        for gamma in gammaArray:
            rbfKernel = svm.RBFKernelWrapper(gamma, 0)
            minDCFArrayRBFSVM1 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=False, Csparam=[-4, 2, 7], kernel=rbfKernel)
            minDCFArrayRBFSVM2 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, Cfp], nFolds, [11, 9, 8], znorm=True, Csparam=[-4, 2, 7], kernel=rbfKernel)
            minDCFArrayRBFSVM3 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=False, Csparam=[-4, 2, 7], kernel=rbfKernel)
            minDCFArrayRBFSVM4 = svm.trainSVMClassifiers(DTR, LTR, [prior, Cfn, 1], nFolds, [11, 9, 8], znorm=True, Csparam=[-4, 2, 7], kernel=rbfKernel)
            
            if saveResults:
                helpers.saveMinDCF("LinSVM", minDCFarrayLinSVM1, effPrior, znorm)
                helpers.saveMinDCF("LinSVM", minDCFarrayLinSVM2, effPrior, True)
                helpers.saveMinDCF("LinSVM", minDCFarrayLinSVM3, 0.5, False)
                helpers.saveMinDCF("LinSVM", minDCFarrayLinSVM4, 0.5, True)
                
                helpers.saveMinDCF(f"PolySVM_d{d}", minDCFarrayPolySVM1, effPrior, znorm)
                helpers.saveMinDCF(f"PolySVM_d{d}", minDCFarrayPolySVM2, effPrior, True)
                helpers.saveMinDCF(f"PolySVM_d{d}", minDCFarrayPolySVM3, 0.5, False)
                helpers.saveMinDCF(f"PolySVM_d{d}", minDCFarrayPolySVM4, 0.5, True)
                
                helpers.saveMinDCF(f"RBFSVM_gamma{gamma}", minDCFArrayRBFSVM1, effPrior, znorm)
                helpers.saveMinDCF(f"RBFSVM_gamma{gamma}", minDCFArrayRBFSVM2, effPrior, True)
                helpers.saveMinDCF(f"RBFSVM_gamma{gamma}", minDCFArrayRBFSVM3, 0.5, False)
                helpers.saveMinDCF(f"RBFSVM_gamma{gamma}", minDCFArrayRBFSVM4, 0.5, True)

        polyK = svm.polyKernelWrapper(1, 2, 0)
        minDCFPolySVMPTS = svm.trainSingleSVMClassifier(DTR, LTR, [prior, Cfn, 0.5], nFolds, 8, C=10e-2, znorm=True, pTs=[0.05, 0.5, 0.9], kernel=polyK)
        rbfK = svm.RBFKernelWrapper(0.001, 0)
        minDCFRBFSVMPTs = svm.trainSingleSVMClassifier(DTR, LTR, [prior, Cfn, 0.5], nFolds, 8, C=10, znorm=False, pTs=[0.05, 0.5, 0.9], kernel=rbfK)
        
        if saveResults:
            helpers.saveMinDCF("PolySVMBest", minDCFPolySVMPTS, 0.5, True)
            helpers.saveMinDCF("RBFSVMBest", minDCFRBFSVMPTs, 0.5, True)
        
        if showResults:
            
            LinSVMResults = np.load(f"results/npy/minDCFLinSVM_prior{formattedPrior}_Znorm{False}.npy")
            zLinSVMResults = np.load(f"results/npy/minDCFLinSVM_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [LinSVMResults, zLinSVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["Linear SVM", "Z-normed LinSVM"], f'Linear SVM minDCFs - effPrior: {formattedPrior}', "C", range(11, 7, -1))
            
            polySVMResults = np.load(f"results/npy/minDCFPolySVM_d2_prior0.09_Znorm{False}.npy")
            zPolySVMResults = np.load(f"results/npy/minDCFPolySVM_d2_prior0.09_Znorm{True}.npy")
            modelsToShow = [polySVMResults, zPolySVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["PolySVM", "Z-normed PolySVM"], f'Polynomial SVM minDCFs - effPrior: 0.09', "C", [11, 9, 8])
            
            polySVMResults = np.load(f"results/npy/minDCFPolySVM_d2_prior0.50_Znorm{False}.npy")
            zPolySVMResults = np.load(f"results/npy/minDCFPolySVM_d2_prior0.50_Znorm{True}.npy")
            modelsToShow = [polySVMResults, zPolySVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["PolySVM", "Z-normed PolySVM"], f'Polynomial SVM minDCFs - effPrior: 0.50', "C", [11, 9, 8])

            rbfSVMREsults3 = np.load("results/npy/minDCFRBFSVM_gamma1e-05_prior0.09_ZnormFalse.npy")
            zrbfSVMREsults3 = np.load("results/npy/minDCFRBFSVM_gamma1e-05_prior0.09_ZnormTrue.npy")
            rbfSVMREsults1 = np.load("results/npy/minDCFRBFSVM_gamma0.001_prior0.09_ZnormFalse.npy")
            zrbfSVMREsults1 = np.load("results/npy/minDCFRBFSVM_gamma0.001_prior0.09_ZnormTrue.npy")
            rbfSVMREsults2 = np.load("results/npy/minDCFRBFSVM_gamma0.1_prior0.09_ZnormFalse.npy")
            zrbfSVMREsults2 = np.load("results/npy/minDCFRBFSVM_gamma0.1_prior0.09_ZnormTrue.npy")
            modelsToShow = [rbfSVMREsults3, zrbfSVMREsults3, rbfSVMREsults1, zrbfSVMREsults1, rbfSVMREsults2, zrbfSVMREsults2]
            vis.plotDifGammaDCFs(
                modelsToShow, 
                ["RBF SVM - gamma 1e-5", "Z-normed RBF SVM - gamma 1e-5", "RBF SVM - gamma 0.001", "Z-normed RBF SVM - gamma 0.001", "RBF SVM - gamma 0.1", "Z-normed RBF SVM - gamma 0.1"],
                f'RBF SVM minDCFs - effPrior: 0.09, PCA: 8', "C", [8])
            

    if runGMM:
        
        workingPoints = [[prior, Cfn, Cfp], [prior, Cfn, 1]]
        
        for workingPoint in workingPoints:
            
            minDCFGMM = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=False, its=4)
            minDCFGMMZ = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=True, its=4)
            
            minDCFGMMTied = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=False, its=4, type="tied")
            minDCFGMMTiedZ = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=True, its=4, type="tied")
            
            minDCFGMMDiag = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=False, its=4, type="diag")
            minDCFGMMDiagZ = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=True, its=4, type="diag")
            
            minDCFGMMTiedDiag = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=False, its=4, type="tiedDiag")
            minDCFGMMTiedDiagZ = gmm.trainAllGMMClassifiers(DTR, LTR, workingPoint, nFolds, pcaDirs=[11, 10, 9, 8], znorm=True, its=4, type="tiedDiag")
            
            prior2, Cfn2, Cfp2 = workingPoint
            effPrior = (prior2*Cfn2)/(prior2*Cfn2 + (1 - prior2)*Cfp2)
            if saveResults:
                helpers.saveMinDCF("GMM", minDCFGMM, effPrior, False)
                helpers.saveMinDCF("GMM", minDCFGMMZ, effPrior, True)
                
                helpers.saveMinDCF("GMMTied", minDCFGMMTied, effPrior, False)
                helpers.saveMinDCF("GMMTied", minDCFGMMTiedZ, effPrior, True)
                
                helpers.saveMinDCF("GMMDiag", minDCFGMMDiag, effPrior, False)
                helpers.saveMinDCF("GMMDiag", minDCFGMMDiagZ, effPrior, True)
                
                helpers.saveMinDCF("GMMTiedDiag", minDCFGMMTiedDiag, effPrior, False)
                helpers.saveMinDCF("GMMTiedDiag", minDCFGMMTiedDiagZ, effPrior, True)
        
        if showResults:
            gmmData = np.load("results/npy/minDCFGMM_prior0.09_ZnormFalse.npy")
            zGmmData = np.load("results/npy/minDCFGMM_prior0.09_ZnormTrue.npy")
            vis.plotGMM_BarChart(gmmData, zGmmData, [11, 10, 9, 8], ["Raw GMM", "Z-normed GMM"], f'GMM minDCFs - effPrior: 0.09')
            
            gmmTiedData = np.load("results/npy/minDCFGMMTied_prior0.09_ZnormFalse.npy")
            zGmmTiedData = np.load("results/npy/minDCFGMMTied_prior0.09_ZnormTrue.npy")
            vis.plotGMM_BarChart(gmmTiedData, zGmmTiedData, [11, 10, 9, 8], ["Raw Tied GMM", "Z-normed Tied GMM"], f'Tied GMM minDCFs - effPrior: 0.09')
            
            gmmDiagData = np.load("results/npy/minDCFGMMDiag_prior0.09_ZnormFalse.npy")
            zGmmDiagData = np.load("results/npy/minDCFGMMDiag_prior0.09_ZnormTrue.npy")
            vis.plotGMM_BarChart(gmmDiagData, zGmmDiagData, [11, 10, 9, 8], ["Raw Diag GMM", "Z-normed Diag GMM"], f'Diag GMM minDCFs - effPrior: 0.09')
            
            gmmTiedDiagData = np.load("results/npy/minDCFGMMTiedDiag_prior0.09_ZnormFalse.npy")
            zGmmTiedDiagData = np.load("results/npy/minDCFGMMTiedDiag_prior0.09_ZnormTrue.npy")
            vis.plotGMM_BarChart(gmmTiedDiagData, zGmmTiedDiagData, [11, 10, 9, 8], ["Raw Tied Diag GMM", "Z-normed Tied Diag GMM"], f'Tied Diag GMM minDCFs - effPrior: 0.09')
            
            
    
    if runCalibration:
        
        calFuse.calibrateAndPlotBestModels(DTR, LTR, [[prior, Cfn, Cfp]])
        calFuse.fuseAndCalibrateGMMandOtherModelsFromResults()
        
    
    # only load testdata here
    DTE, LTE = loader.loadData('data/Test.txt')
    
    if runEvaluation:
        
        #testing.evaluate(DTR, LTR, DTE, LTE)
        #testing.evaluateFusions()
        gmmTestScores = np.load('results/npy/testing/gmmScoresTest_uncalibratedScores.npy')
        svmTestScores = np.load('results/npy/testing/svmScoresTest_uncalibratedScores.npy')
        logRegTestScores = np.load('results/npy/testing/logRegScoresTest_uncalibratedScores.npy')
        gmmTestLabels = np.load('results/npy/testing/gmmScoresTest_uncalibratedLabels.npy')
        svmTestLabels = np.load('results/npy/testing/svmScoresTest_uncalibratedLabels.npy')
        logRegTestLabels = np.load('results/npy/testing/logRegScoresTest_uncalibratedLabels.npy')
        # vis.plotDET([gmmTestScores, logRegTestScores, svmTestScores], gmmTestLabels, ["GMM", "Q-LR", "poly-SVM"], ['r', 'b', 'g'], 'testing/best3DET')
        #testing.calibrateScores()
        
        # testing.compareLogRegTrainTest(DTR, LTR, DTE, LTE)
        # testing.compareSVMTrainTest(DTR, LTR, DTE, LTE, svm.polyKernelWrapper(1, 2, 0), 'poly', firstZnorm=True, firstPCA=8, secondC=10e-2)
        testing.compareSVMTrainTest(DTR, LTR, DTE, LTE, svm.RBFKernelWrapper(0.001, 0), 'rbf', firstZnorm=False, firstPCA=8, secondC=10)
        
        testing.compareGMMTrainTest(DTR, LTR, DTE, LTE)