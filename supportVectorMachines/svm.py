import numpy as np
import scipy.optimize as spoptim
import helpers.helpers as helpers


class LinearSVMClassifier:
    def __init__(self, trainData, trainLabels) -> None:
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
    
    
    def computePrimalObjective(self, wStar, C):
        
        primalLoss = 0
        regularizer = 0.5 * (np.linalg.norm(wStar)**2)
        Z = helpers.vrow(2 * self.trainLabels - 1)

        rightPart = 1 - Z*np.dot(wStar.T, self.trainDataHat)
        leftPart = np.zeros(rightPart.shape[1])
        
        stackedForMaxOperation = np.vstack((leftPart, rightPart))
        
        sumPart = C*np.sum(np.max(stackedForMaxOperation, axis=0))
        
        primalLoss = regularizer + sumPart
        return primalLoss
    
    
    def train(self, C: float, k: float = 1):
        Carray = np.ones((self.numSamples, ))*C
        zeros = np.zeros(self.numSamples,)
        boxConstraints = np.asarray(list(zip(zeros, Carray)))
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
    
    
    def predict(self, testData, wStar, threshold: float = 0, k: float = 1):
        
        w = wStar[:-1]
        b = wStar[-1]
        predictions = np.zeros(testData.shape[1])
        
        score = np.dot(w.T, testData)+b*k
        predictions = score > threshold
    
        return score, predictions
        


def polyKernelWrapper(c, d, ksi):

    def polyKernel(x1, x2):
        
        result = (np.dot(x1.T, x2) + c)**d + np.sqrt(ksi)
        
        return result
    
    return polyKernel


def RBFKernelWrapper(gamma, ksi):
    
    def radialBiasFunctionKernel(x1, x2):
        
        result = np.empty((x1.shape[1],x2.shape[1]))
        
        for i in range(x1.shape[1]):
            for j in range(x2.shape[1]):
                result[i,j] = np.exp(-gamma*np.linalg.norm(x1[:,i]-x2[:,j])**2)

        return result + np.sqrt(ksi)
    
    return radialBiasFunctionKernel


class KernelSVMClassifier:
    def __init__(self, trainData, trainLabels) -> None:
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
    
    
    def train(self, C: float, kernelFunc):
        
        Carray = np.ones((self.numSamples, ))*C
        zeros = np.zeros(self.numSamples,)
        boxConstraints = np.asarray(list(zip(zeros, Carray)))
        
        Z = helpers.vrow(2 * self.trainLabels - 1)
        self.HHat = Z * Z.T * kernelFunc(self.trainData, self.trainData)
        
        x0 = np.zeros(self.numSamples)   
        
        # dualLoss is neg bc w comput LHatD not JHatD!!
        alphaStar, dualLoss, info = spoptim.fmin_l_bfgs_b(self.computeDualObjective, x0, bounds=boxConstraints, factr=1.0, maxfun=100000, maxiter=100000)
        
        return alphaStar, dualLoss
    
    
    def predict(self, testData, alphaStar, threshold, kernelFunc):
        
        Z = helpers.vcol(2 * self.trainLabels - 1)
                
        score = np.sum(helpers.vcol(alphaStar) * Z * kernelFunc(self.trainData, testData) , axis=0)
        predictions = score > threshold
    
        return predictions