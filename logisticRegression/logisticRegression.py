import numpy as np
import scipy.optimize as spoptim

import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import evaluation.evaluation as eval

class logRegClassifier:
    """
    This class implements a BINARY Logistic Regression Classifier
    """
    def __init__(self, DTR : np.ndarray, LTR: np.ndarray, l: float, pi: float = None) -> None:
        """Init function to initialize the object with the required parameters

        Args:
            DTR (np.ndarray): train dataset
            LTR (np.ndarray): train labels
            l (float): lambda that serves as regularizer term
        """
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.pi = pi
        self.K = len(set(self.LTR)) # number of classes
        self.D = DTR.shape[0] # dimensionality / number of features
        self.nSamples = DTR.shape[1]
    
    
    def logregBinary(self, v : np.ndarray) -> float:
        """
        Implements the binary logistic regression function.
        Args:
            v (np.ndarray): the model parameters joined into 1 vector

        Returns:
            float: the return of the logisitic regression function
        """
        w = v[0: -1]
        b = v[-1]
        
        regularizer = self.l / 2 * (np.linalg.norm(w)**2)
        
        if self.pi == None:
            H = 0
            
            for i in range(self.nSamples):
                z = 2*self.LTR[i] - 1
                curval = np.logaddexp(0, -z*(np.dot(w.T, self.DTR[:, i]) + b))
                H += curval
            
            J = regularizer + H / self.nSamples
            
            return J
        
        else:
            risk1 = 0
            risk0 = 0
            Nt = np.sum(self.LTR == 1)
            Nf = np.sum(self.LTR == 0)
            classT = self.DTR[:, self.LTR == 1]
            classF = self.DTR[:, self.LTR == 0]
            
            for i in range(classT.shape[1]):
                risk1 += np.logaddexp(0, -1 * (np.dot(w.T, classT[:, i]) + b))
            for i in range(classF.shape[1]):
                risk0 += np.logaddexp(0, 1 * (np.dot(w.T, classF[:, i]) + b))
            
            H = self.pi/Nt*risk1 + (1 - self.pi)/Nf*risk0
            J = regularizer + H
            
            return J
    
    def trainBin(self):
        """
        Trains the model according to the logistic regression function.
        Minimizes the logreg based on the model parameteres using the l_bfgs_b function
        Returns:
            wtrain (np.ndarray): logreg model params
            btrain (np.ndarray): logreg model params
        """
        x0 = np.zeros(self.D + 1)        
        inLog, outLog, extra = spoptim.fmin_l_bfgs_b(self.logregBinary, x0, approx_grad=True)
        
        wtrain = inLog[0:-1]
        btrain = inLog[-1]

        return wtrain, btrain


    def evaluateBin(self, DTE: np.ndarray, w: np.ndarray, b: np.ndarray, thr: float = 0) -> np.ndarray:
        """Evaluates the model on DTE test dataset, based on the model parameters. Returns the predicted labels.

        Args:
            DTE (np.ndarray): Test data
            w (np.ndarray): w model param of logistic regression
            b (np.ndarray): b model param of logistic regression
            thr (float, optional): Threshold, based on which the classification happens either to class 0 or 1. Defaults to 0.

        Returns:
            np.ndarray: predicted labels
        """
        predictions = np.zeros(DTE.shape[1])
        score = np.dot(w.T, DTE) + b
        predictions = score > thr
        
        return score, predictions
    
def transformTrainAndTestToQuadratic(DTR: np.ndarray, DTE: np.ndarray) -> np.ndarray:
    """Transforms the train and test data to quadratic form

    Args:
        DTR (np.ndarray): Train data
        DTE (np.ndarray): Test data

    Returns:
        np.ndarray: Transformed train data
        np.ndarray: Transformed test data
    """
    qDTR = np.zeros((DTR.shape[0] ** 2 + DTR.shape[0], DTR.shape[1]))
    qDTE = np.zeros((DTR.shape[0] ** 2 + DTR.shape[0], DTE.shape[1]))

    for i in range(DTR.shape[1]):
        qDTR[:, i:i + 1] = stackArray(DTR[:, i:i + 1])
    for i in range(DTE.shape[1]):
        qDTE[:, i:i + 1] = stackArray(DTE[:, i:i + 1])

    return qDTR, qDTE


def stackArray(array: np.ndarray) -> np.ndarray:
    """
    Stacks the array in a quadratic form

    Args:
        array (np.ndarray): array to be stacked

    Returns:
        np.ndarray: stacked array
    """
    
    n_f = array.shape[0]
    xx_t = np.dot(array, array.T)
    column = np.zeros((n_f ** 2 + n_f, 1))
    for i in range(n_f):
        column[i * n_f:i * n_f + n_f, :] = xx_t[:, i:i + 1]
    column[n_f ** 2: n_f ** 2 + n_f, :] = array
    
    return column


def trainLogRegClassifiers(startPCA: int, endPCA: int, DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, nFolds: int, znorm: bool, quadratic: bool) -> np.ndarray:

    prior, Cfn, Cfp = workingPoint
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    
    minDCFarray = []
    
    for j in range(startPCA, endPCA, -1):
        
        # no pca
        if(j == 11):
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)
            kdata, klabels = helpers.splitToKFold(DTR, LTR, K=nFolds)
        else:
            if znorm:
                DTR, _, _ = preproc.zNormalization(DTR)

            reducedData, _ = preproc.computePCA(DTR, j)
            kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
            
        llrsLogReg = []
        
        lambdas = np.logspace(-6, 4, 11)
        
        for l in range(len(lambdas)):
            curLambda = lambdas[l]
            
            correctEvalLabels = []
            
            llrsLogReg.append([curLambda, []])
            
            for i in range(0, nFolds):
                
                trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
                correctEvalLabels.append(evalLabels)

                
                # training  Logistic Regression
                if quadratic:
                    trainingData, evalData = transformTrainAndTestToQuadratic(trainingData, evalData)
                
                qLogRegObj = logRegClassifier(DTR=trainingData, LTR=trainingLabels, l=curLambda, pi=effPrior)
                qwtrain, qbtrain = qLogRegObj.trainBin()
                qlogScores, _ = qLogRegObj.evaluateBin(DTE=evalData, w=qwtrain, b=qbtrain)
                
                llrsLogReg[l][1].append(qlogScores)

            
        correctEvalLabels = np.hstack(correctEvalLabels)
        for i in range(len(llrsLogReg)):
            
            llrsLogReg[i][1] = np.hstack(llrsLogReg[i][1])
            minDCFLogReg = eval.computeMinDCF(llrsLogReg[i][1], correctEvalLabels, prior, Cfn, Cfp)
            minDCFarray.append([j, llrsLogReg[i][0], minDCFLogReg])
            
    return minDCFarray


def trainSingleLogRegClassifier(DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, PCADir: int, l: float, znorm: bool, quadratic: bool, pTs: list, nFolds: int, mode: str = None) -> np.ndarray:
    
    prior, Cfn, Cfp = workingPoint
    minDCFarrayLogReg = []
    llrsLogReg = []
    
    if mode == 'calibration':
        scores = []
    
    if znorm: 
        DTR, _, _ = preproc.zNormalization(DTR)
    if PCADir == 11:
        reducedData = DTR
    else:
        reducedData, _ = preproc.computePCA(DTR, PCADir)
    kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
    
    for pT in pTs:
        
        llrsLogReg.append([pT, []])
        correctEvalLabels = []
        
        for i in range(0, nFolds):
            
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
            correctEvalLabels.append(evalLabels)
            
            # training Linear Logistic Regression candidate A
            if quadratic:
                trainingData, evalData = transformTrainAndTestToQuadratic(trainingData, evalData)
            
            if mode == 'calibration':
                # in case of calibration specific prior weight is not used because it never outperformed the default during the training phase
                logRegObj9 = logRegClassifier(DTR=trainingData, LTR=trainingLabels, l=l)
            else:
                logRegObj9 = logRegClassifier(DTR=trainingData, LTR=trainingLabels, l=l, pi=pT)
            
            wtrain9, btrain9 = logRegObj9.trainBin()
            logScores9, _ = logRegObj9.evaluateBin(DTE=evalData, w=wtrain9, b=btrain9)
            llrsLogReg[-1][1].append(logScores9)
            
            if mode == 'calibration':
                scores.append(logScores9)
            
    if mode == 'calibration':
        return scores, correctEvalLabels
    
    correctEvalLabels = np.hstack(correctEvalLabels)
    for i in range(len(llrsLogReg)):
        
        llrsLogReg[i][1] = np.hstack(llrsLogReg[i][1])
        minDCFLogReg8 = eval.computeMinDCF(llrsLogReg[i][1], correctEvalLabels, prior, Cfn, Cfp)
        minDCFarrayLogReg.append([llrsLogReg[i][0], minDCFLogReg8])

    return minDCFarrayLogReg