import numpy as np
import logisticRegression.logisticRegression as logreg
import supportVectorMachines.svm as svm
import gaussianMixtureModels.gmm as gmm
import visualization.visualization as vis
import helpers.helpers as helpers
import calibrationFusion.calibrationFusion as calFuse
import evaluation.evaluation as eval
import gaussianMixtureModels.gmm as gmm

def evaluate(DTR, LTR, DTE, LTE):
    
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


def compareLogRegTrainTest(DTR, LTR, DTE, LTE):
    
    # first for the lambda
    
    lambdaValues = np.logspace(-5, 1, 7)
    minDCFsDev = []
    minDCFsTest = []
    
    for l in lambdaValues:
        
        scoresDev, labelsDev = logreg.trainSingleLogRegClassifier(DTR, LTR, [0.5, 1, 10], 11, l, znorm=True, quadratic=True, pTs=[None], nFolds=5, mode='calibration')
        scoresDev = np.hstack(scoresDev)
        labelsDev = np.hstack(labelsDev)
        minDCFlogregDev = eval.computeMinDCF(scoresDev, labelsDev, 0.5, 1, 10)
        
        scoresEval = logreg.trainSingleLogRegOnFullTrainData(DTR, LTR, DTE, l, znorm=True, pcaDir=11)
        minDCFlogregTest = eval.computeMinDCF(scoresEval, LTE, 0.5, 1, 10)
        
        minDCFsDev.append([11, l, minDCFlogregDev])
        minDCFsTest.append([11, l, minDCFlogregTest])
    
    np.savetxt("results/txt/testing/logRegDCFsDev_lambda", minDCFsDev)
    np.savetxt("results/txt/testing/logRegDCFsTest_lambda", minDCFsTest)
    vis.plotLogRegDCFs([minDCFsDev, minDCFsTest], ['Train', 'Test'], f'LogReg DCFs comparison for Dev and Test data', 'lambda', [11])
    
    for znorm in [True, False]:
        
        minDCFsDev = []
        minDCFsTest = []
        for pcaDir in [11, 9, 8]:
            
            # lambda is the best one from the previous part
            scoresDev, labelsDev = logreg.trainSingleLogRegClassifier(DTR, LTR, [0.5, 1, 10], pcaDir, 10e-2, znorm=znorm, quadratic=True, pTs=[None], nFolds=5, mode='calibration')
            scoresDev = np.hstack(scoresDev)
            labelsDev = np.hstack(labelsDev)
            minDCFlogregDev = eval.computeMinDCF(scoresDev, labelsDev, 0.5, 1, 10)
            
            scoresEval = logreg.trainSingleLogRegOnFullTrainData(DTR, LTR, DTE, 10e-2, znorm=znorm, pcaDir=pcaDir)
            minDCFlogregTest = eval.computeMinDCF(scoresEval, LTE, 0.5, 1, 10)
            
            minDCFsDev.append([pcaDir, 10e-3, minDCFlogregDev])
            minDCFsTest.append([pcaDir, 10e-3, minDCFlogregTest])
        
        np.savetxt(f"results/txt/testing/logRegDCFsDev_znorm{znorm}", minDCFsDev)
        np.savetxt(f"results/txt/testing/logRegDCFsTest_znorm{znorm}", minDCFsTest)
        np.save(f"results/npy/testing/logRegDCFsDev_znorm{znorm}", minDCFsDev)
        np.save(f"results/npy/testing/logRegDCFsTest_znorm{znorm}", minDCFsTest)


def compareSVMTrainTest(DTR, LTR, DTE, LTE, kernel, kernelName, firstZnorm, firstPCA, secondC):
    
    # first for the C of poly kernel
    CValues = np.logspace(-5, 2, 8)
    minDCFsDev = []
    minDCFsTest = []
    
    for C in CValues:
        
        scoresDev, labelsDev = svm.trainSingleSVMClassifier(DTR, LTR, [0.5, 1, 10], nFolds=5, PCADir=firstPCA, C=C, znorm=firstZnorm, pTs=[None], kernel=kernel, mode='calibration')
        scoresDev = np.hstack(scoresDev)
        labelsDev = np.hstack(labelsDev)
        minDCFsvmDev = eval.computeMinDCF(scoresDev, labelsDev, 0.5, 1, 10)
        
        scoresEval = svm.trainSingleKernelSVMOnFullTrainData(DTR, LTR, DTE, c=C, pcaDir=firstPCA, prior=None, znorm=firstZnorm, kernel=kernel)
        minDCFsvmTest = eval.computeMinDCF(scoresEval, LTE, 0.5, 1, 10)
        
        minDCFsDev.append([8, C, minDCFsvmDev])
        minDCFsTest.append([8, C, minDCFsvmTest])
    
    np.savetxt(f"results/txt/testing/{kernelName}svmDCFsDev_C", minDCFsDev)
    np.savetxt(f"results/txt/testing/{kernelName}svmDCFsTest_C", minDCFsTest)
    vis.plotLogRegDCFs([minDCFsDev, minDCFsTest], ['Train', 'Test'], f'{kernelName}-SVM DCFs comparison for Dev and Test data', 'C', [8])
    
    for znorm in [True, False]:
        
        minDCFsTest = []
        
        for pca in [11, 9, 8]:
            
            scoresEval = svm.trainSingleKernelSVMOnFullTrainData(DTR, LTR, DTE, c=secondC, pcaDir=pca, prior=None, znorm=znorm, kernel=kernel)
            minDCFSVMTest = eval.computeMinDCF(scoresEval, LTE, 0.5, 1, 10)
            
            minDCFsTest.append([pca, 10e-2, minDCFSVMTest])
            
        np.savetxt(f"results/txt/testing/{kernelName}svmDCFsTest_znorm{znorm}", minDCFsTest)


def evaluateGMMsOnTest(DTR, LTR, DTE, LTE):
    
    gmmScoresTD11 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=False, its=3, type='tiedDiag')
    gmmScoresTD10 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=False, its=3, type='tiedDiag')
    gmmScoresTD09 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=False, its=3, type='tiedDiag')
    gmmScoresTD08 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=False, its=3, type='tiedDiag')
    
    gmmScoresTD11Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=True, its=3, type='tiedDiag')
    gmmScoresTD10Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=True, its=3, type='tiedDiag')
    gmmScoresTD09Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=True, its=3, type='tiedDiag')
    gmmScoresTD08Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=True, its=3, type='tiedDiag')
    
    
    gmmScoresD11 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=False, its=3, type='diag')
    gmmScoresD10 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=False, its=3, type='diag')
    gmmScoresD09 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=False, its=3, type='diag')
    gmmScoresD08 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=False, its=3, type='diag')
    
    gmmScoresD11Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=True, its=3, type='diag')
    gmmScoresD10Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=True, its=3, type='diag')
    gmmScoresD09Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=True, its=3, type='diag')
    gmmScoresD08Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=True, its=3, type='diag')
    
    
    gmmScoresT11 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=False, its=3, type='tied')
    gmmScoresT10 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=False, its=3, type='tied')
    gmmScoresT09 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=False, its=3, type='tied')
    gmmScoresT08 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=False, its=3, type='tied')
    
    gmmScoresT11Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=True, its=3, type='tied')
    gmmScoresT10Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=True, its=3, type='tied')
    gmmScoresT09Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=True, its=3, type='tied')
    gmmScoresT08Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=True, its=3, type='tied')
    
    
    gmmScoresF11 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=False, its=3)
    gmmScoresF10 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=False, its=3)
    gmmScoresF09 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=False, its=3)
    gmmScoresF08 = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=False, its=3)
    
    gmmScoresF11Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=11, znorm=True, its=3)
    gmmScoresF10Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=10, znorm=True, its=3)
    gmmScoresF09Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=9, znorm=True, its=3)
    gmmScoresF08Z = gmm.trainBestGMMClassifierOnFullTrainData(DTR, LTR, DTE, pcaDir=8, znorm=True, its=3)
    
    gmmMinDCFsTD11 = []
    gmmMinDCFsTD10 = []
    gmmMinDCFsTD09 = []
    gmmMinDCFsTD08 = []
    gmmMinDCFsTD11Z = []
    gmmMinDCFsTD10Z = []
    gmmMinDCFsTD09Z = []
    gmmMinDCFsTD08Z = []
    gmmMinDCFsD11 = []
    gmmMinDCFsD10 = []
    gmmMinDCFsD09 = []
    gmmMinDCFsD08 = []
    gmmMinDCFsD11Z = []
    gmmMinDCFsD10Z = []
    gmmMinDCFsD09Z = []
    gmmMinDCFsD08Z = []
    gmmMinDCFsT11 = []
    gmmMinDCFsT10 = []
    gmmMinDCFsT09 = []
    gmmMinDCFsT08 = []
    gmmMinDCFsT11Z = []
    gmmMinDCFsT10Z = []
    gmmMinDCFsT09Z = []
    gmmMinDCFsT08Z = []
    gmmMinDCFsF11 = []
    gmmMinDCFsF10 = []
    gmmMinDCFsF09 = []
    gmmMinDCFsF08 = []
    gmmMinDCFsF11Z = []
    gmmMinDCFsF10Z = []
    gmmMinDCFsF09Z = []
    gmmMinDCFsF08Z = []
    
    for g in range(len(gmmScoresD08)):
        
        gmmMinDCFsTD11.append(eval.computeMinDCF(gmmScoresTD11[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD10.append(eval.computeMinDCF(gmmScoresTD10[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD09.append(eval.computeMinDCF(gmmScoresTD09[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD08.append(eval.computeMinDCF(gmmScoresTD08[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD11Z.append(eval.computeMinDCF(gmmScoresTD11Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD10Z.append(eval.computeMinDCF(gmmScoresTD10Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD09Z.append(eval.computeMinDCF(gmmScoresTD09Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsTD08Z.append(eval.computeMinDCF(gmmScoresTD08Z[g], LTE, 0.5, 1, 10))
        
        gmmMinDCFsD11.append(eval.computeMinDCF(gmmScoresD11[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD10.append(eval.computeMinDCF(gmmScoresD10[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD09.append(eval.computeMinDCF(gmmScoresD09[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD08.append(eval.computeMinDCF(gmmScoresD08[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD11Z.append(eval.computeMinDCF(gmmScoresD11Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD10Z.append(eval.computeMinDCF(gmmScoresD10Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD09Z.append(eval.computeMinDCF(gmmScoresD09Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsD08Z.append(eval.computeMinDCF(gmmScoresD08Z[g], LTE, 0.5, 1, 10))
        
        gmmMinDCFsT11.append(eval.computeMinDCF(gmmScoresT11[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT10.append(eval.computeMinDCF(gmmScoresT10[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT09.append(eval.computeMinDCF(gmmScoresT09[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT08.append(eval.computeMinDCF(gmmScoresT08[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT11Z.append(eval.computeMinDCF(gmmScoresT11Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT10Z.append(eval.computeMinDCF(gmmScoresT10Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT09Z.append(eval.computeMinDCF(gmmScoresT09Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsT08Z.append(eval.computeMinDCF(gmmScoresT08Z[g], LTE, 0.5, 1, 10))
        
        gmmMinDCFsF11.append(eval.computeMinDCF(gmmScoresF11[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF10.append(eval.computeMinDCF(gmmScoresF10[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF09.append(eval.computeMinDCF(gmmScoresF09[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF08.append(eval.computeMinDCF(gmmScoresF08[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF11Z.append(eval.computeMinDCF(gmmScoresF11Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF10Z.append(eval.computeMinDCF(gmmScoresF10Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF09Z.append(eval.computeMinDCF(gmmScoresF09Z[g], LTE, 0.5, 1, 10))
        gmmMinDCFsF08Z.append(eval.computeMinDCF(gmmScoresF08Z[g], LTE, 0.5, 1, 10))
    
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD11", gmmMinDCFsTD11)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD10", gmmMinDCFsTD10)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD09", gmmMinDCFsTD09)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD08", gmmMinDCFsTD08)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD11Z", gmmMinDCFsTD11Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD10Z", gmmMinDCFsTD10Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD09Z", gmmMinDCFsTD09Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsTD08Z", gmmMinDCFsTD08Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD11", gmmMinDCFsD11)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD10", gmmMinDCFsD10)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD09", gmmMinDCFsD09)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD08", gmmMinDCFsD08)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD11Z", gmmMinDCFsD11Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD10Z", gmmMinDCFsD10Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD09Z", gmmMinDCFsD09Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsD08Z", gmmMinDCFsD08Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT11", gmmMinDCFsT11)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT10", gmmMinDCFsT10)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT09", gmmMinDCFsT09)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT08", gmmMinDCFsT08)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT11Z", gmmMinDCFsT11Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT10Z", gmmMinDCFsT10Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT09Z", gmmMinDCFsT09Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsT08Z", gmmMinDCFsT08Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF11", gmmMinDCFsF11)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF10", gmmMinDCFsF10)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF09", gmmMinDCFsF09)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF08", gmmMinDCFsF08)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF11Z", gmmMinDCFsF11Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF10Z", gmmMinDCFsF10Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF09Z", gmmMinDCFsF09Z)
    np.savetxt(f"results/txt/testing/gmmMinDCFsF08Z", gmmMinDCFsF08Z)
