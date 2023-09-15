import numpy as np
import logisticRegression.logisticRegression as logreg
import supportVectorMachines.svm as svm
import gaussianMixtureModels.gmm as gmm
import visualization.visualization as vis
import helpers.helpers as helpers
import calibrationFusion.calibrationFusion as calFuse
def evaluate(DTR, LTR, DTE, LTE):
    
    #TODO: run the Gaussian too
    
    gmmScores = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=False, its=3, type='tiedDiag')
    gmmScores = gmmScores[-1, :]
    helpers.saveCalAndUncalScoresAndLabels(f'gmmScoresTest', 'testing', gmmScores, LTE)
    vis.plotBayesError(gmmScores, LTE, "GMM Bayes Error Plot", 'testing', None, None)
    
    logRegScores = logreg.trainSingleLogRegOnFullTrainData(DTR, LTR, DTE, 10e-3)
    helpers.saveCalAndUncalScoresAndLabels(f'logRegScoresTest', 'testing', logRegScores, LTE)
    vis.plotBayesError(logRegScores, LTE, "LogisticRegression Bayes Error Plot", 'testing', None, None)
    
    svmScores = svm.trainSingleKernelSVMOnFullTrainData(DTR, LTR, DTE, 10e-2)
    helpers.saveCalAndUncalScoresAndLabels(f'svmScoresTest', 'testing', svmScores, LTE)
    vis.plotBayesError(svmScores, LTE, "SVM Bayes Error Plot", 'testing', None, None)


def evaluateFusions():
    
    logregScores = np.load('results/npy/testing/logRegScoresTest_uncalibratedScores.npy')
    logregLabels = np.load('results/npy/testing/logRegScoresTest_uncalibratedLabels.npy')
    svmScores = np.load('results/npy/testing/svmScoresTest_uncalibratedScores.npy')
    svmLabels = np.load('results/npy/testing/svmScoresTest_uncalibratedLabels.npy')
    gmmScores = np.load('results/npy/testing/gmmScoresTest_uncalibratedScores.npy')
    gmmLabels = np.load('results/npy/testing/gmmScoresTest_uncalibratedLabels.npy')
    
    # GMM SVM fusion
    gmmScoresP, _ = calFuse.permuteFolds(gmmScores, gmmLabels)
    svmScoresP, labels2 = calFuse.permuteFolds(svmScores, svmLabels)
    GMMSVMScores = np.vstack([gmmScoresP, svmScoresP])

    # fused scores
    GMMSVMFusedScores, GMMSVMFusedLabels = logreg.trainSingleLogRegClassifier(DTR=GMMSVMScores, LTR=labels2, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    GMMSVMFusedScores = np.hstack(GMMSVMFusedScores)
    GMMSVMFusedLabels = np.hstack(GMMSVMFusedLabels)
    helpers.saveCalAndUncalScoresAndLabels(f'GMMSVMFused', 'testing', GMMSVMFusedScores, GMMSVMFusedLabels)
    vis.plotBayesError(GMMSVMFusedScores, GMMSVMFusedLabels, 'GMM SVM Fusion model Bayes Error Plot', "testing")
    
    
    # GMM LogReg fusion
    gmmScoresP, _ = calFuse.permuteFolds(gmmScores, gmmLabels)
    logRegScoresP, labels2 = calFuse.permuteFolds(logregScores, logregLabels)
    GMMLogRegScores = np.vstack([gmmScoresP, logRegScoresP])
    
    # fused scores
    GMMLogRegFusedScores, GMMLogRegFusedLabels = logreg.trainSingleLogRegClassifier(DTR=GMMLogRegScores, LTR=labels2, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    GMMLogRegFusedLabels = np.hstack(GMMLogRegFusedLabels)
    GMMLogRegFusedScores = np.hstack(GMMLogRegFusedScores)
    helpers.saveCalAndUncalScoresAndLabels(f'GMMLogRegFused', 'testing', GMMLogRegFusedScores, GMMLogRegFusedLabels)
    vis.plotBayesError(GMMLogRegFusedScores, GMMLogRegFusedLabels, 'GMM LogReg Fusion model Bayes Error Plot', "testing")


def calibrateScores():
    
    logregScores = np.load('results/npy/testing/logRegScoresTest_uncalibratedScores.npy')
    logregLabels = np.load('results/npy/testing/logRegScoresTest_uncalibratedLabels.npy')
    svmScores = np.load('results/npy/testing/svmScoresTest_uncalibratedScores.npy')
    svmLabels = np.load('results/npy/testing/svmScoresTest_uncalibratedLabels.npy')
    
    svmScores, svmLabels = calFuse.permuteFolds(svmScores, svmLabels)
    calibratedSVMScores, calibratedSVMLabels = logreg.trainSingleLogRegClassifier(DTR=np.array([svmScores]), LTR=svmLabels, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    helpers.saveCalAndUncalScoresAndLabels(f'bestSVM', 'testing', None, None, np.hstack(calibratedSVMScores), np.hstack(calibratedSVMLabels))
    vis.plotBayesError(svmScores, svmLabels, 'Poly-SVM Bayes Error Plot with calibrated results', "testing", np.hstack(calibratedSVMScores), np.hstack(calibratedSVMLabels))

    logregScores, logregLabels = calFuse.permuteFolds(logregScores, logregLabels)
    calibratedLogRegScores, calibratedLabels = logreg.trainSingleLogRegClassifier(DTR=np.array([logregScores]), LTR=logregLabels, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    helpers.saveCalAndUncalScoresAndLabels(f'bestLogReg', 'testing', None, None, np.hstack(calibratedLogRegScores), np.hstack(calibratedLabels))
    vis.plotBayesError(logregScores, logregLabels, 'Logistic Regression Bayes Error Plot with calibrated results', "testing", np.hstack(calibratedLogRegScores), np.hstack(calibratedLabels))
