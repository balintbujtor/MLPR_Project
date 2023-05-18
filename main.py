import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import generativeModels.generativeModels as generativeModels

if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')
    DTE, LTE = loader.loadData('data/Test.txt')
    
    priorProb = helpers.vcol(np.ones(2)/2.0)
    _, sPost = generativeModels.logMVG(DTR, LTR, DTE, 2, priorProb)
    predLabelsMVG = np.argmax(sPost, 0)
    accuracyMVG = helpers.compute_accuracy(predLabelsMVG, LTE)
    
    print("finished")