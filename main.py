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
    formattedPrior = "{:.2f}".format(effPrior)

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

        print('trained')
        
        pTs = [0.05, 0.5, 0.9]

        minDCFarrayLogReg8 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-5, znorm=True, quadratic=False, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogReg9 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=9, l=10e-2, znorm=False, quadratic=False, pTs=pTs, nFolds=nFolds)
        
        minDCFarrayLogRegQ8 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=8, l=10e-2, znorm=False, quadratic=True, pTs=pTs, nFolds=nFolds)
        minDCFarrayLogRegQ10 = logReg.trainBestLogRegClassifierWDiffPriorWeights(DTR, LTR, [prior, Cfn, Cfp], PCADir=11, l=10e-3, znorm=True, quadratic=True, pTs=pTs, nFolds=nFolds)
        
        if saveResults:
            
            helpers.saveMinDCF("LogRegPCA8_lambda10e-5", minDCFarrayLogReg8, effPrior, zNorm=True)
            helpers.saveMinDCF("LogRegPCA9_lambda10e-2", minDCFarrayLogReg9, effPrior, zNorm=False)
            
            helpers.saveMinDCF("QLogRegPCA8_lambda10e-2", minDCFarrayLogRegQ8, effPrior, zNorm=False)
            helpers.saveMinDCF("QLogRegPCA10_lambda10e-3", minDCFarrayLogRegQ10, effPrior, zNorm=True)

    if runSVM:

        minDCFarrayLinSVM = svm.trainLinSVMClassifiers(11, 7, DTR, LTR, [prior, Cfn, Cfp], nFolds, znorm, "linear")

        if saveResults:
            helpers.saveMinDCF("LinSVM", minDCFarrayLinSVM, effPrior, znorm)

        if showResults:
            
            LinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{False}.npy")
            zLinSVMResults = np.load(f"results/npy/minDCFLogReg_prior{formattedPrior}_Znorm{True}.npy")
            modelsToShow = [LinSVMResults, zLinSVMResults]
            vis.plotLogRegDCFs(modelsToShow, ["Linear SVM", "Z-normed LinSVM"], f'Linear SVM minDCFs - effPrior: {formattedPrior}', "C", range(11, 7, -1))
