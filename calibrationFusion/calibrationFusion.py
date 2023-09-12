import numpy as np

import visualization.visualization as vis

from helpers.helpers import saveCalAndUncalScoresAndLabels
from supportVectorMachines.svm import trainSingleSVMClassifier, polyKernelWrapper
from gaussianMixtureModels.gmm import trainAllGMMClassifiers
from logisticRegression.logisticRegression import trainSingleLogRegClassifier

def permuteFolds(scores: list, labels: list) -> np.ndarray:
    
    np.random.seed(0)
    idx = np.random.permutation(len(scores))
    scores = np.array(scores)[idx]
    labels = np.array(labels)[idx]
    scores = np.hstack(scores)
    labels = np.hstack(labels)
    
    return scores, labels

def calibrateAndPlotBestModels(DTR: np.ndarray, LTR: np.ndarray, workingPoints: list) :
    
    for wp in workingPoints:    

        # GMM
        gmmScores, gmmLabels = trainAllGMMClassifiers(DTR=DTR, LTR=LTR, workingPoint=wp, nFolds=5, pcaDirs=[11], znorm=False, its=3, type='tiedDiag', mode='calibration')
        # keep only the scores of the 8 components GMM (last one)
        gmmScores = gmmScores[:, -1, :]
        gmmScores, gmmLabels = permuteFolds(gmmScores, gmmLabels)
        calibratedGMMScores, calibratedGMMLabels = trainSingleLogRegClassifier(DTR=np.array([gmmScores]), LTR=gmmLabels, workingPoint=wp, PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
        saveCalAndUncalScoresAndLabels(f'bestGMM', gmmScores, gmmLabels, np.hstack(calibratedGMMScores), np.hstack(calibratedGMMLabels))
        vis.plotBayesError(gmmScores, gmmLabels, 'Calibrated and non-calibrated GMM', np.hstack(calibratedGMMScores), np.hstack(calibratedGMMLabels))
        
        # SVM
        polyK = polyKernelWrapper(1, 2, 0)
        svmScores, svmLabels = trainSingleSVMClassifier(DTR=DTR, LTR=LTR, workingPoint= wp, nFolds=5, PCADir=8, C=10e-2, znorm=True, pTs=[None], kernel=polyK, mode='calibration')
        svmScores, svmLabels = permuteFolds(svmScores, svmLabels)
        calibratedSVMScores, calibratedSVMLabels = trainSingleLogRegClassifier(DTR=np.array([svmScores]), LTR=svmLabels, workingPoint=wp, PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
        saveCalAndUncalScoresAndLabels(f'bestSVM', svmScores, svmLabels, np.hstack(calibratedSVMScores), np.hstack(calibratedSVMLabels))
        vis.plotBayesError(svmScores, svmLabels, 'Calibrated and non-calibrated Poly-SVM', np.hstack(calibratedSVMScores), np.hstack(calibratedSVMLabels))
        
        # logisctic regression
        logRegScores, logRegLabels = trainSingleLogRegClassifier(DTR=DTR, LTR=LTR, workingPoint=wp, PCADir=11, l=10e-3, znorm=True, quadratic=True, pTs=[None], nFolds=5, mode='calibration')
        logRegScores, logRegLabels = permuteFolds(logRegScores, logRegLabels)
        calibratedLogRegScores, calibratedLabels = trainSingleLogRegClassifier(DTR=np.array([logRegScores]), LTR=logRegLabels, workingPoint=wp, PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
        saveCalAndUncalScoresAndLabels(f'bestLogReg', logRegScores, logRegLabels, np.hstack(calibratedLogRegScores), np.hstack(calibratedLabels))
        vis.plotBayesError(logRegScores, logRegLabels, 'Calibrated and non-calibrated Logistic Regression', np.hstack(calibratedLogRegScores), np.hstack(calibratedLabels))


def fuseAndCalibrateGMMandOtherModelsFromResults():
    
    # fuse GMM and SVM scores
    gmmScores = np.load('results/npy/calibration/bestGMM_uncalibratedScores.npy')
    logRegScores = np.load('results/npy/calibration/bestLogReg_uncalibratedScores.npy')
    svmScores = np.load('results/npy/calibration/bestSVM_uncalibratedScores.npy')
    labels = np.load('results/npy/calibration/bestGMM_uncalibratedLabels.npy')
    
    # GMM SVM fusion
    gmmScoresP, _ = permuteFolds(gmmScores, labels)
    svmScoresP, labels2 = permuteFolds(svmScores, labels)
    GMMSVMScores = np.vstack([gmmScoresP, svmScoresP])

    # fused scores
    GMMSVMFusedScores, GMMSVMFusedLabels = trainSingleLogRegClassifier(DTR=GMMSVMScores, LTR=labels2, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    # calibrated scores
    GMMSVMFusedScores, GMMSVMFusedLabels = permuteFolds(GMMSVMFusedScores, GMMSVMFusedLabels)
    calibratedGMMSVMFusedScores, calibratedGMMSVMFusedLabels = trainSingleLogRegClassifier(DTR=np.array([GMMSVMFusedScores]), LTR=GMMSVMFusedLabels, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    saveCalAndUncalScoresAndLabels(f'GMMSVMFused', GMMSVMFusedScores, GMMSVMFusedLabels, np.hstack(calibratedGMMSVMFusedScores), np.hstack(calibratedGMMSVMFusedLabels))
    vis.plotBayesError(GMMSVMFusedScores, GMMSVMFusedLabels, 'Calibrated and non-calibrated GMM SVM Fusion model', np.hstack(calibratedGMMSVMFusedScores), np.hstack(calibratedGMMSVMFusedLabels))
    
    
    # GMM LogReg fusion
    gmmScoresP, _ = permuteFolds(gmmScores, labels)
    logRegScoresP, labels2 = permuteFolds(logRegScores, labels)
    GMMLogRegScores = np.vstack([gmmScoresP, logRegScoresP])
    
    # fused scores
    GMMLogRegFusedScores, GMMLogRegFusedLabels = trainSingleLogRegClassifier(DTR=GMMLogRegScores, LTR=labels2, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    # calibrated scores
    GMMLogRegFusedScores, GMMLogRegFusedLabels = permuteFolds(GMMLogRegFusedScores, GMMLogRegFusedLabels)
    calibratedGMMLogRegFusedScores, calibratedGMMLogRegFusedLabels = trainSingleLogRegClassifier(DTR=np.array([GMMLogRegFusedScores]), LTR=GMMLogRegFusedLabels, workingPoint=[0.5, 1, 10], PCADir=11, l=0, znorm=False, quadratic=False, pTs=[None], nFolds=5, mode='calibration')
    saveCalAndUncalScoresAndLabels(f'GMMLogRegFused', GMMLogRegFusedScores, GMMLogRegFusedLabels, np.hstack(calibratedGMMLogRegFusedScores), np.hstack(calibratedGMMLogRegFusedLabels))
    vis.plotBayesError(GMMLogRegFusedScores, GMMLogRegFusedLabels, 'Calibrated and non-calibrated GMM LogReg Fusion model', np.hstack(calibratedGMMLogRegFusedScores), np.hstack(calibratedGMMLogRegFusedLabels))