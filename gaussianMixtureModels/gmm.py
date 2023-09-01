import scipy.special
import numpy as np
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import evaluation.evaluation as eval


def computeDensScores(X: np.ndarray, gmm: np.ndarray):
    
    logDensScores = np.empty((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        logDensScores[g, :] = helpers.logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) + np.log(gmm[g][0])
            
    return logDensScores


def constrainEigenValues(covMat, psi):
    
    U, s, _ = np.linalg.svd(covMat)
    s[s<psi] = psi
    newCovMat = np.dot(U, helpers.vcol(s)*U.T)
    
    return newCovMat


def GMMDiagonal(myGmm: list):
    for comp in range(len(myGmm)):
        myGmm[comp][2] = myGmm[comp][2]*np.eye(myGmm[comp][2].shape[0])

    return myGmm


def GMMTied(myGmm: list, Zg: np.ndarray, X: np.ndarray):
    tiedCov = np.sum([(Zg[g]*myGmm[g][2])for g in range(len(myGmm))],axis=0)/ X.shape[1]
    
    for comp in range(len(myGmm)):
        myGmm[comp][2] = tiedCov                

    return myGmm


def EM_Algorithm(X: np.ndarray, gmmIn: list, threshold: float = 1e-6, psi: float = 0.01, type: str = None):
    
    myGmm = gmmIn
    avgLL = None
    newAvgLL = None
    
    
    while(avgLL is None or np.abs(newAvgLL - avgLL) > threshold):
        
        avgLL = newAvgLL
        
        # Expectation step
        logSJoint = np.zeros((len(myGmm), X.shape[1]))
        for g in range(len(myGmm)):
            logSJoint[g, :] = helpers.logpdf_GAU_ND(X, myGmm[g][1], myGmm[g][2]) + np.log(myGmm[g][0])
        
        logMarginal = scipy.special.logsumexp(logSJoint, axis=0)
        
        newAvgLL = sum(logMarginal)/X.shape[1]

        logSPost = logSJoint - logMarginal
        # same as sPost
        sPost = np.exp(logSPost)
        
        myNewGmm = []
        
    
        Zg=helpers.vcol(sPost.sum(axis=1))
        Fg=np.dot(sPost,X.T)
        Sg=[np.dot(sPost[i]*X,X.T) for i in range(len(myGmm))]
        
        #new GMM parameters
        munext=Fg/Zg
        Cnext=[Sg[i]/Zg[i]-np.dot(helpers.vcol(munext[i]),helpers.vrow(munext[i])) for i in range(len(myGmm))]
        wnext=Zg/Zg.sum(axis=0)[0]
        
        for i in range (len(wnext)):
            myNewGmm.append([wnext[i], helpers.vcol(munext[i]),Cnext[i]])        
        
        # if true then Diagonal GMMs (covariance mat) is used
        if type == 'diag':
            myNewGmm = GMMDiagonal(myNewGmm)
            
        # if true, then tied model is used
        elif type == 'tied':
            myNewGmm = GMMTied(myNewGmm, Zg, X)
        
        elif type == 'tiedDiag':
            myNewGmm = GMMTied(myNewGmm, Zg, X)
            myNewGmm = GMMDiagonal(myNewGmm)
            
        # constrain the eigenvalues
        for i in range(len(myGmm)):
            myNewGmm[i][2] = constrainEigenValues(myNewGmm[i][2], psi)

        #update the gmm params
        myGmm = myNewGmm
        
    print('final precision: ', np.abs(newAvgLL - avgLL))
    
    return myGmm


def splitGMM(GMMToSplit, alpha):

    newGMM = []
    
    for i in range(len(GMMToSplit)):
        U, s, _ = np.linalg.svd(GMMToSplit[i][2])
        displacementVect = U[:, 0:1] * s[0]**0.5 * alpha
        
        newGMM.append((GMMToSplit[i][0]/2, GMMToSplit[i][1] + displacementVect, GMMToSplit[i][2]))
        newGMM.append((GMMToSplit[i][0]/2, GMMToSplit[i][1] - displacementVect, GMMToSplit[i][2]))

    return newGMM


def LBG_Algorithm(dataset: np.ndarray, gmmInit: list, its: int, alpha: float = 0.1, psi: float = 0.01, type : str = None):
    
    print(f'Running LBG algorithm to split initial components into {2**its} and {type} covariance matrix')
    
    
    if len(gmmInit) == 1:
        gmm = gmmInit
        if type == 'diag':
            gmm = GMMDiagonal(gmmInit)
            
        elif type == 'tied':
            gmm = GMMTied(gmmInit, [dataset.shape[1]], dataset)
            
        elif type == 'tiedDiag':
            gmm = GMMTied(gmmInit, [dataset.shape[1]], dataset)
            gmm = GMMDiagonal(gmm)
        
        for i in range(len(gmm)):
            gmm[i][2] = constrainEigenValues(gmm[i][2], psi)
        
        startGMM = EM_Algorithm(dataset, gmm, psi=psi, type=type)
    
    else:
        startGMM = gmmInit
        
    for i in range(its):
        
        print('iteration #', i + 1)
        splittedGMM = splitGMM(startGMM, alpha)
        
        startGMM = EM_Algorithm(dataset, splittedGMM, psi=psi, type=type)   
    
    return startGMM    



def computeGMM_LLR(evalSet: np.ndarray, gmm: list):
    
    llrs = []
    
    for i in range(2):
        
        score = []
        classGMM = gmm[i]
        
        for j in range(len(classGMM)):
            score.append(helpers.logpdf_GAU_ND(evalSet, classGMM[j][1], classGMM[j][2]) + np.log(classGMM[j][0]))
        
        score = np.vstack(score)
        logDens = scipy.special.logsumexp(score, axis=0)
        
        llrs.append(logDens)
    
    llr = llrs[1] - llrs[0]
    
    return llr



def trainAllGMMClassifiers(DTR: np.ndarray, LTR: np.ndarray, workingPoint: list, nFolds: int, pcaDirs: list, znorm: bool, its: int, type: str = None) -> np.ndarray:

    prior, Cfn, Cfp = workingPoint
    
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

            reducedData, _ = preproc.computePCA(DTR, dim)
            kdata, klabels = helpers.splitToKFold(reducedData, LTR, K=nFolds)
            
        llrs = []
        
        correctEvalLabels = []
        
        
        for i in range(0, nFolds):
            
            llrFold = []
            
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)                
            correctEvalLabels.append(evalLabels)
            
            gmmFold = []
            
            # for each class
            for classes in range(2):
                
                # first split to have 2 components for the class
                mu, cov = helpers.ML_estimates(trainingData[:, trainingLabels==classes])
                gmmFold.append(LBG_Algorithm(trainingData[:, trainingLabels == classes], [[1, mu, cov]], 1, type=type))

                # then split its - 1 times to have 2**(its) components for the class
                for j in range(1, its):
                    gmmFold.append(LBG_Algorithm(trainingData[:, trainingLabels == classes], gmmFold[j-1], 1, type=type))
            
            # we have 2*2**(its) components in total (2 classes)
            for g in range(len(gmmFold)//2):
                gmmFoldPerComponent = []
                gmmFoldPerComponent.append(gmmFold[g])
                gmmFoldPerComponent.append(gmmFold[ len(gmmFold)//2 + g])
                
                llrGMMComp = computeGMM_LLR(evalData, gmmFoldPerComponent)
                
                llrFold.append(llrGMMComp)
                
            
            llrs.append(llrFold)
        
        # join the llrs of each fold
        llrsNP = np.array(llrs)
        llrsNP = np.hstack(llrsNP)
        correctEvalLabels = np.hstack(correctEvalLabels)

        # compute the minDCF for each GMM with different components
        for i in range(its):
            minDCF = eval.computeMinDCF(llrsNP[i], correctEvalLabels, prior, Cfn, Cfp)
            minDCFarray.append([dim, 2**(i + 1), minDCF])       
        
    return minDCFarray