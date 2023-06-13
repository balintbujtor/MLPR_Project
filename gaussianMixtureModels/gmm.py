import scipy.special
import numpy as np
import helpers.helpers as helpers

def logpdfGMM(X: np.ndarray, gmm: np.ndarray, computeJoint: bool = True):
    
    logDensScores = np.empty((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        compLogDens = helpers.logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        logDensScores[g, :] = compLogDens
        
        if(computeJoint):
            logDensScores[g, :] += np.log(gmm[g][0])
            
    return logDensScores

def computelogMarginalGMM(logDensScores: np.ndarray):
    return scipy.special.logsumexp(logDensScores, axis=0)

def constrainEigenValues(covMat, psi):
    
    U, s, _ = np.linalg.svd(covMat)
    s[s<psi] = psi
    newCovMat = np.dot(U, helpers.vcol(s)*U.T)
    
    return newCovMat


def EM_Algorithm(X: np.ndarray, gmm: list, threshold: float, psi: float = 0, diagonal: bool = False, tied: bool = False):
    
    myGmm = gmm
    avgLL = 0
    stoppingCriterion = 100
    
    while(stoppingCriterion > threshold):
        # Expectation step
        logJointScores = logpdfGMM(X, myGmm)        
        logMarginalDensities = computelogMarginalGMM(logJointScores)
        logSPost = logJointScores - logMarginalDensities
        # same as sPost
        gamma = np.exp(logSPost)
        
        # Maximization step
        Zg = helpers.vcol(np.sum(gamma, axis=1))
        Fg = np.dot(gamma, X.T)
        Sg = [np.dot(gamma[i]*X,X.T) for i in range(len(myGmm))]
        
        # compute the criterion that decides whether to stop
        newAvgLL = sum(logMarginalDensities)/X.shape[1]
        print(newAvgLL)
        stoppingCriterion = np.abs(newAvgLL - avgLL)
        avgLL = newAvgLL
        
        # obtain the new params
        # shape: number of features * number of components
        wNew = Zg / np.sum(Zg, axis=0)
        muNew = Fg / Zg
        covNew = [ Sg[i] / Zg[i] - np.dot(helpers.vcol(muNew[i]), helpers.vrow(muNew[i])) for i in range(len(myGmm))]
        
        # if true then Diagonal GMMs (covariance mat) is used
        # TODO: verify if works
        if(diagonal):
            covNew = [covNew[i]*np.eye(covNew[i].shape[0]) for i in range(len(myGmm))]
            
        # if true, then tied model is used
        # TODO: verify if works
        if(tied):
            tiedCov = np.sum([(Zg[g]*covNew[g])for g in range(len(myGmm))],axis=0)/ X.shape[1]
            covNew = [tiedCov for i in range(len(myGmm))]
            
        covNew = [constrainEigenValues(covNew[i], psi) for i in range(len(myGmm))]

        #update the gmm params
        myGmm = [(wNew[i], muNew[i], covNew[i]) for i in range(len(myGmm))]
        
    return myGmm


def splitGMM(GMMToSplit, alpha):

    newGMM = []
    
    for i in range(len(GMMToSplit)):
        U, s, _ = np.linalg.svd(GMMToSplit[i][2])
        displacementVect = U[:, 0:1] * s[0]**0.5 * alpha
        
        newGMM.append((GMMToSplit[i][0]/2, GMMToSplit[i][1] + displacementVect, GMMToSplit[i][2]))
        newGMM.append((GMMToSplit[i][0]/2, GMMToSplit[i][1] - displacementVect, GMMToSplit[i][2]))

    return newGMM


def LBG_Algorithm(dataset: np.ndarray, its: int, alpha: float, threshold: float, psi : float = 0):
    
    print('Running LBG algorithm')
    
    mu1, cov1 = helpers.ML_estimates(dataset)
    cov1 = constrainEigenValues(cov1, psi)
    
    GMMs = [(1.0, mu1, cov1)]
    GMMi_EMs = []
    GMMi_EMs.append(GMMs)
    
    for i in range(its):
        
        print('iteration #', i)
        GMMs = splitGMM(GMMs, alpha)
        GMMi_EM = EM_Algorithm(dataset, GMMs, threshold, psi)
        GMMi_EMs.append(GMMi_EM)
    
    return GMMi_EMs