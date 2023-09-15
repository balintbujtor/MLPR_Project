import numpy as np
import scipy.optimize as spoptim
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import evaluation.evaluation as eval

def computeBalancedCArray(labels: np.ndarray, C: float, pT: float):
    
    zeros = np.zeros(labels.shape[0])
    CArray = np.zeros(labels.shape[0])
    empiricalPT = np.sum(labels == 1) / labels.shape[0]
    CArray[labels == 1] = C * pT / empiricalPT
    CArray[labels == 0] = C * (1 - pT) / (1 - empiricalPT)
    boxConstraints = np.asarray(list(zip(zeros, CArray)))
    
    return boxConstraints

class LinearSVMClassifier:
    def __init__(self, trainData: np.ndarray, trainLabels: np.ndarray) -> None:
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.numClasses = len(set(self.trainLabels)) # number of classes
        self.dim = trainData.shape[0] # dimensionality / number of features
        self.numSamples = trainData.shape[1]
    
    
    def computeDualObjective(self, alpha : np.ndarray):
        identityVector = np.ones(self.numSamples)
        LHat = 0.5*np.dot(alpha.T, np.dot(self.HHat, alpha)) - np.dot(alpha.T, identityVector)
        
        LHatGradient = (np.dot(self.HHat, alpha) - 1)
        
        return LHat, LHatGradient
    
    
    def computePrimalObjective(self, wStar: np.ndarray, C: float):
        
        primalLoss = 0
        regularizer = 0.5 * (np.linalg.norm(wStar)**2)
        Z = helpers.vrow(2 * self.trainLabels - 1)

        rightPart = 1 - Z*np.dot(wStar.T, self.trainDataHat)
        leftPart = np.zeros(rightPart.shape[1])
        
        stackedForMaxOperation = np.vstack((leftPart, rightPart))
        
        sumPart = C*np.sum(np.max(stackedForMaxOperation, axis=0))
        
        primalLoss = regularizer + sumPart
        return primalLoss
    
    
    def train(self, C: float, pT: float = None, k: float = 1):
        
        if pT == None:
            Carray = np.ones((self.numSamples, ))*C
            zeros = np.zeros(self.numSamples,)
            boxConstraints = np.asarray(list(zip(zeros, Carray)))
            
        else:
            boxConstraints = computeBalancedCArray(self.trainLabels, C, pT)
            
        Karray = np.ones((1, self.numSamples))*k
        self.trainDataHat = np.vstack([self.trainData, Karray])
        
        GHat = np.dot(self.trainDataHat.T, self.trainDataHat)
        Z = helpers.vrow(2 * self.trainLabels - 1)
        self.HHat = Z * Z.T * GHat
        
        x0 = np.zeros(self.numSamples)   
        
        # dualLoss is neg bc w comput LHatD not JHatD!!
        alphaStar, dualLoss, info = spoptim.fmin_l_bfgs_b(self.computeDualObjective, x0, bounds=boxConstraints, factr=1.0)
        
        wStar = np.sum(alphaStar * Z * self.trainDataHat, axis=1)
        
        primalLoss = self.computePrimalObjective(wStar, C)

        return alphaStar, wStar, primalLoss, dualLoss
    
    
    def predict(self, testData: np.ndarray, wStar: np.ndarray, threshold: float = 0, k: float = 1):
        
        w = wStar[:-1]
        b = wStar[-1]
        predictions = np.zeros(testData.shape[1])
        
        score = np.dot(w.T, testData)+b*k
        predictions = score > threshold
    
        return score, predictions
        


def polyKernelWrapper(c: float, d: float, ksi: float):

    def polyKernel(x1: np.ndarray, x2: np.ndarray):
        
        result = (np.dot(x1.T, x2) + c)**d + np.sqrt(ksi)
        
        return result
    
    return polyKernel


def RBFKernelWrapper(gamma: float, ksi: float):
    
    def radialBasisFunctionKernel(x1: np.ndarray, x2: np.ndarray):
        
        result = np.empty((x1.shape[1],x2.shape[1]))
        
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                result[i,j] = np.exp(-gamma*np.linalg.norm(x1[:,i]-x2[:,j])**2)

        return result + np.sqrt(ksi)
    
    return radialBasisFunctionKernel

# TODO: refactor SVM to have only one class
class KernelSVMClassifier:
    def __init__(self, trainData: np.ndarray, trainLabels: np.ndarray) -> None:
        self.trainData = trainData
        self.trainLabels = trainLabels
        self.numClasses = len(set(self.trainLabels)) # number of classes
        self.dim = trainData.shape[0] # dimensionality / number of features
        self.numSamples = trainData.shape[1]
    
    
    def computeDualObjective(self, alpha : np.ndarray):
        identityVector = np.ones(self.numSamples)
        LHat = 0.5*np.dot(np.dot(alpha.T, self.HHat), alpha) - np.dot(alpha.T, identityVector)
        
        LHatGradient = (np.dot(self.HHat, alpha) - 1)
        
        return LHat, LHatGradient
    
    
    def train(self, C: float, kernelFunc, pT: float = None):
        
        if pT == None:
            Carray = np.ones((self.numSamples, ))*C
            zeros = np.zeros(self.numSamples,)
            boxConstraints = np.asarray(list(zip(zeros, Carray)))
            
        else:
            boxConstraints = computeBalancedCArray(self.trainLabels, C, pT)
        
        Z = helpers.vrow(2 * self.trainLabels - 1)
        self.HHat = Z * Z.T * kernelFunc(self.trainData, self.trainData)
        
        x0 = np.zeros(self.numSamples)   
        
        # dualLoss is neg bc we compute LHatD not JHatD!!
        alphaStar, dualLoss, info = spoptim.fmin_l_bfgs_b(self.computeDualObjective, x0, bounds=boxConstraints, factr=1.0)
        
        return alphaStar, dualLoss
    
    
    def predict(self, testData, alphaStar, kernelFunc, threshold: float = 0):
        
        Z = helpers.vcol(2 * self.trainLabels - 1)
                
        scores = np.sum(helpers.vcol(alphaStar) * Z * kernelFunc(self.trainData, testData) , axis=0)
        predictions = scores > threshold
    
        return scores, predictions


def trainSVMClassifiers(DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, nFolds: int, pcaDirs: list, znorm: bool, Csparam: list = None, kernel = None) -> np.ndarray:

    prior, Cfn, Cfp = workingPoint
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    
    minDCFarray = []
    
    for dim in pcaDirs:
        
        # no pca
        if(dim == 11):
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)
            kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
        else:
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)

            reducedData, _, _ = preproc.computePCA(DTR, dim)
            kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
            
        llrsSVM = []
        
        if Csparam == None:
            Cs = np.logspace(-6, 4, 11)
        else:
            Cs = np.logspace(Csparam[0], Csparam[1], Csparam[2])
            
        for c in range(len(Cs)):
            
            curC = Cs[c]
            correctEvalLabels = []
            llrsSVM.append([curC, []])
            
            for i in range(0, nFolds):
                
                trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                correctEvalLabels.append(evalLabels)
                
                if kernel == None:
                    # training Linear SVM, without class rebalancing
                    linSVMObj = LinearSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                    linAlpha, linW, linPrimal, linDual = linSVMObj.train(curC)
                    linLogScores, linPreds = linSVMObj.predict(evalData, linW)
                    
                    llrsSVM[c][1].append(linLogScores)
                
                else:
                    # training non-linear SVM without class rebalancing
                    poliSVMObj = KernelSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                    
    
                    alphaStar, dualLoss = poliSVMObj.train(curC, kernelFunc=kernel)
                    poliLogScores, poliPreds = poliSVMObj.predict(evalData, alphaStar, kernelFunc=kernel)

                    #print("Incorrect kernel function for training SVM")
                    #return None
                    
                    llrsSVM[c][1].append(poliLogScores)
                
        correctEvalLabels = np.hstack(correctEvalLabels)
        for i in range(len(llrsSVM)):
            
            llrsSVM[i][1] = np.hstack(llrsSVM[i][1])
            minDCFLinSVM = eval.computeMinDCF(llrsSVM[i][1], correctEvalLabels, prior, Cfn, Cfp)
            minDCFarray.append([dim, llrsSVM[i][0], minDCFLinSVM])
    
    return minDCFarray


def trainSingleSVMClassifier(DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, nFolds: int, PCADir: int, C: float, znorm: bool, pTs: list, kernel = None, mode: str = None) -> np.ndarray:
    
    prior, Cfn, Cfp = workingPoint
    minDCFarray = []
    llrs = []
    
    if mode == 'calibration':
        scores = []
    
    if znorm: 
        DTR, _, _ = preproc.zNormalization(DTR)
    reducedData, _, _ = preproc.computePCA(DTR, PCADir)
    kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
    
    for pT in pTs:
        
        llrs.append([pT, []])
        correctEvalLabels = []
        
        for i in range(0, nFolds):
            
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
            correctEvalLabels.append(evalLabels)
            
            # training single SVM
            if kernel == None:
                # training Linear SVM, without class rebalancing
                linSVMObj = LinearSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                if mode == 'calibration':
                    # in case of calibration specific prior weight is not used because it never outperformed the default during the training phase
                    linAlpha, linW, linPrimal, linDual = linSVMObj.train(C)
                else:
                    linAlpha, linW, linPrimal, linDual = linSVMObj.train(C, pT=pT)
                    
                linLogScores, linPreds = linSVMObj.predict(evalData, linW)
                
                llrs[-1][1].append(linLogScores)
                
                if mode == 'calibration':
                    scores.append(linLogScores)
            
            else:
                # training non-linear SVM without class rebalancing
                poliSVMObj = KernelSVMClassifier(trainData=trainingData, trainLabels=trainingLabels)
                
                if mode == 'calibration':
                    # in case of calibration specific prior weight is not used because it never outperformed the default during the training phase
                    alphaStar, dualLoss = poliSVMObj.train(C, kernelFunc=kernel)
                else:
                    alphaStar, dualLoss = poliSVMObj.train(C, kernelFunc=kernel, pT=pT)

                poliLogScores, poliPreds = poliSVMObj.predict(evalData, alphaStar, kernelFunc=kernel)
                llrs[-1][1].append(poliLogScores)     
                
                if mode == 'calibration':
                    scores.append(poliLogScores)
    
    if mode == 'calibration':
        return scores, correctEvalLabels
    
    correctEvalLabels = np.hstack(correctEvalLabels)
    
    for i in range(len(llrs)):
        
        llrs[i][1] = np.hstack(llrs[i][1])
        minDCF = eval.computeMinDCF(llrs[i][1], correctEvalLabels, prior, Cfn, Cfp)
        minDCFarray.append([llrs[i][0], minDCF])

    return minDCFarray


def trainSingleKernelSVMOnFullTrainData(DTR, LTR, DTE, c, pcaDir: int = 8, prior: float = None):

    DTR, mean, stdDev = preproc.zNormalization(DTR)
    DTE, _, _ = preproc.zNormalization(DTE, mean, stdDev)
    
    reducedTrainData, covTrain, _ = preproc.computePCA(DTR, pcaDir)
    reducedTestData, _, _ = preproc.computePCA(DTE, pcaDir, covTrain)
    
    polyKernel = polyKernelWrapper(1, 2, 0)
    # training non-linear SVM without class rebalancing
    poliSVMObj = KernelSVMClassifier(trainData=reducedTrainData, trainLabels=LTR)
    
    if prior == None:
        alphaStar, dualLoss = poliSVMObj.train(c, kernelFunc=polyKernel)
    else:
        alphaStar, dualLoss = poliSVMObj.train(c, kernelFunc=polyKernel, pT=prior)
    
    poliLogScores, poliPreds = poliSVMObj.predict(reducedTestData, alphaStar, kernelFunc=polyKernel)

    return poliLogScores