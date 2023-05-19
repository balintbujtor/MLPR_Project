import numpy as np
import scipy.optimize as spoptim

class logRegClassifier:
    """
    This class implements a BINARY Logistic Regression Classifier
    """
    def __init__(self, DTR : np.ndarray, LTR: np.ndarray, l: float) -> None:
        """Init function to initialize the object with the required parameters

        Args:
            DTR (np.ndarray): train dataset
            LTR (np.ndarray): train labels
            l (float): lambda that serves as regularizer term
        """
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
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
        H = 0
        
        for i in range(self.nSamples):
            z = 2*self.LTR[i] - 1
            curval = np.logaddexp(0, -z*(np.dot(w.T, self.DTR[:, i]) + b))
            H += curval
        
        J = regularizer + H / self.nSamples
        
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
        
        return predictions