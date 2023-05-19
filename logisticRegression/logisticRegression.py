import numpy as np
import scipy.optimize as spoptim

class logRegClassifier:
    """_summary_
    """
    def __init__(self, DTR, LTR, l) -> None:
        
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.K = len(set(self.LTR)) # number of classes
        self.D = DTR.shape[0] # dimensionality / number of features
        self.nSamples = DTR.shape[1]
        
    def logregBinary(self, v : np.ndarray):
        
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
        x0 = np.zeros(self.D + 1)        
        inLog, outLog, extra = spoptim.fmin_l_bfgs_b(self.logregBinary, x0, approx_grad=True)
        
        wtrain = inLog[0:-1]
        btrain = inLog[-1]

        return wtrain, btrain


    def evaluateBin(self, DTE, w, b, thr = 0):
        
        predictions = np.zeros(DTE.shape[1])
        score = np.dot(w.T, DTE) + b
        predictions = score > thr
        
        return predictions