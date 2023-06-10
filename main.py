import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import generativeModels.generativeModels as generativeModels
import logisticRegression.logisticRegression as logReg
import matplotlib.pyplot as plot
if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')
    DTE, LTE = loader.loadData('data/Test.txt')
    
    PCAdirs = preproc.computePCA(DTR, 2)
    
    class0 = PCAdirs[:, LTR == 0]
    class1 = PCAdirs[:, LTR == 1]
    
    class0attr0 = class0[0, :]
    class1attr0 = class1[0, :]

    plot.hist(class0attr0, bins=25, density=True, fc=(0, 0, 1, 0.5))
    plot.hist(class1attr0, bins=25, density=True, fc=(1, 0, 0, 0.5))
    plot.legend(["class false", "class true"])
    plot.show()
    
    LDAdirs = preproc.computeLDA(DTR, LTR, 2)
    
    
    
    
    
    
    
    priorProb = helpers.vcol(np.ones(2)/2.0)
    _, sPost = generativeModels.logMVG(DTR, LTR, DTE, 2, priorProb)
    predLabelsMVG = np.argmax(sPost, 0)
    accuracyMVG = helpers.compute_accuracy(predLabelsMVG, LTE)
    
    logRegressor = logReg.logRegClassifier(DTR, LTR, 0.01)
    w, b = logRegressor.trainBin()
    preds = logRegressor.evaluateBin(DTE, w, b)
    accLogReg = helpers.compute_accuracy(preds, LTE)
    
    #kfold
    # TODO: check k fold split, how it should behave correctly
    predLabelsMVG = []
    nsplits, splitbounds = helpers.splitToKFold(DTR.shape[1], 5)

    for iteration in range(0, nsplits):
        (DTRcur, LTRcur), (DTEcur, LTEcur) = helpers.getCurrentKFoldSplit(DTR, LTR, iteration, splitbounds)
        
        # MVG with K-fold
        sJointLogMVG, sPostLogMVG = generativeModels.logMVG(DTRcur, LTRcur, DTEcur, 2, priorProb)
        predLabelsMVG.append(np.argmax(sPostLogMVG, 0))
    
    predLabelsMVG = np.asarray(predLabelsMVG).flatten()
    accuracyMVG = helpers.compute_accuracy(predLabelsMVG, LTE)    

    print("finished")