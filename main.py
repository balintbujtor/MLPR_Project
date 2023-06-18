import numpy as np
import loader.loader as loader
import helpers.helpers as helpers
import preprocessing.preprocessing as preproc
import generativeModels.generativeModels as generativeModels
import logisticRegression.logisticRegression as logReg
import matplotlib.pyplot as plot
import visualization.visualization as vis
import evaluation.evaluation as eval


visualize = False

if __name__ == "__main__":

    DTR, LTR = loader.loadData('data/Train.txt')
    DTE, LTE = loader.loadData('data/Test.txt')

    # initial preprocessing
    
    PCAdirs, _ = preproc.computePCA(DTR, 2)
    pcaData9Dirs, _ = preproc.computePCA(DTR, 9) #PCA with 9 directions
    pcaData10Dirs, _ = preproc.computePCA(DTR, 10) #PCA with 10 directions
    
    pcaClass0 = PCAdirs[:, LTR == 0]
    pcaClass1 = PCAdirs[:, LTR == 1]
    pcaClass0attr0 = pcaClass0[0, :]
    pcaClass1attr0 = pcaClass1[0, :]
    pcaClass0attr1 = pcaClass0[1, :]
    pcaClass1attr1 = pcaClass1[1, :]
    cumRatios = preproc.computeCumVarRatios(DTR)
    
    LDAdir = preproc.computeLDA(DTR, LTR, 1)
    ldaClass0 = LDAdir[:, LTR == 0]
    ladClass1 = LDAdir[:, LTR == 1]
    ldaClass0attr0 = ldaClass0[0, :]
    ldaClass1attr0 = ladClass1[0, :] 
    
    if visualize:
        vis.plotHistogram(pcaClass0attr0, pcaClass1attr0)
        vis.plotHistogram(pcaClass0attr1, pcaClass1attr1)
        vis.plotScatter(pcaClass0, pcaClass1)
        vis.plotHistogram(ldaClass0attr0, ldaClass1attr0)
        vis.plotCorrMat(DTR[:, LTR == 0])
        vis.plotCorrMat(DTR[:, LTR == 1])
        vis.plotCorrMat(DTR)
        vis.plotCumVarRatios(cumRatios, DTR.shape[0] + 1)


    # setting up working point and eff prior
    prior = 0.5
    Cfn = 1
    Cfp = 10
    effPrior = (prior*Cfn)/(prior*Cfn + (1 - prior)*Cfp)
    priorProb = np.asarray([[effPrior], [1 - effPrior]])
    
    
    #kfold
    postlogscores = []
    correctEvalLabels = []
    nFolds = 5
    
    def modelPCADir(modelName, numberOfDir):
        pcaDataNDirs, _ = preproc.computePCA(DTR, numberOfDir) #PCA with n directions
        
        postlogscores = []
        correctEvalLabels = []
        nFolds = 5
        kdata, klabels = helpers.splitToKFold(pcaDataNDirs, LTR) #Change DTR with pcaData9Dirs
        
        for i in range(0, nFolds):
        
            trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
        
            _, sPostLogMVG = modelName(trainingData, trainingLabels, evalData, 2, priorProb)
            postlogscores.append(sPostLogMVG)
            correctEvalLabels.append(evalLabels)
            
        postlogscores = np.hstack(postlogscores)
        llr = np.log(postlogscores[1] / postlogscores[0])
        
        predLabels = np.argmax(postlogscores, 0)
        correctEvalLabels = np.hstack(correctEvalLabels)
        
        acc = eval.compute_accuracy(predLabels, correctEvalLabels)
        minDCF = eval.computeMinDCF(llr, correctEvalLabels, prior, Cfn, Cfp)

        print(minDCF)
        return minDCF
    
    #Function that calls generativeModel with K-fold
    def generativeModelCall(modelName):
        _, sPostLogMVG = modelName(trainingData, trainingLabels, evalData, 2, priorProb)
        postlogscores.append(sPostLogMVG)
        correctEvalLabels.append(evalLabels)
     
    
    pcaData8Dirs, _ = preproc.computePCA(DTR, 8) #PCA with 8 directions

    kdata, klabels = helpers.splitToKFold(DTR, LTR) #Change DTR with pcaData9Dirs
    

    #------------------------------------------------------   Test to call and get minDFC for logreg model
    for fold in range(nFolds):
        # Get the training and validation data for the current fold
        trainData, trainLabels, valData, valLabels = helpers.getCurrentKFoldSplit(kdata, klabels, fold, nFolds)

        # Create and train the logistic regression classifier
        classifier = logReg.logRegClassifier(trainData, trainLabels, l=0.1)  # Use appropriate regularization parameter
        wtrain, btrain = classifier.trainBin()  # Train the classifier and obtain the optimized model parameters

        # Compute the post-log likelihood scores and evaluation labels for the validation data
        postlogscores_fold = []
        evalLabels_fold = []
            
        for i in range(valData.shape[1]):
            postlogscore = classifier.logregBinary(np.concatenate((valData[:, i], [1])))
            postlogscores_fold.append(postlogscore)
            evalLabel = classifier.evaluateBin(valData[:, i].reshape(-1, 1), wtrain, btrain, thr=0)

            evalLabels_fold.append(evalLabel)

        # Store the performance scores and evaluation labels for the current fold
        postlogscores.append(postlogscores_fold)
        correctEvalLabels.append(evalLabels_fold)

    # Compute the minDCF using the stored performance scores and evaluation labels
    minDCF = eval.computeMinDCF(postlogscores, correctEvalLabels, prior, Cfn, Cfp)
    print("logreg: ")
    print("minDCF:", minDCF)
    #-----------------------------------------

    for i in range(0, nFolds):
        
        trainingData, trainingLabels, evalData, evalLabels = helpers.getCurrentKFoldSplit(kdata, klabels, i, nFolds)
        
        # MVG with K-fold
        generativeModelCall(generativeModels.logMVG)
        #_, sPostLogMVG = generativeModels.logMVG(trainingData, trainingLabels, evalData, 2, priorProb)
        #postlogscores.append(sPostLogMVG)
        #correctEvalLabels.append(evalLabels)
        
        # tied
        #----------logTiedMVG
        # MVG with K-fold
        #_, sPostLogMVG = generativeModels.logTiedMVG(trainingData, trainingLabels, evalData, 2, priorProb)
        #postlogscores.append(sPostLogMVG)
        #correctEvalLabels.append(evalLabels)
        
        # naive
        #----------logNaiveBayes
        
        
        #naivetied
        #----------logTiedNaiveBayes
        

        #logreg
        #----------logRegClassifier
        
        #TODO function
        
        #logreg znorm
        
        #?????
        #----------logregBinary

    postlogscores = np.hstack(postlogscores)
    llr = np.log(postlogscores[1] / postlogscores[0])
    
    predLabels = np.argmax(postlogscores, 0)
    correctEvalLabels = np.hstack(correctEvalLabels)
    
    acc = eval.compute_accuracy(predLabels, correctEvalLabels)
    minDCFMVg = eval.computeMinDCF(llr, correctEvalLabels, prior, Cfn, Cfp)

    print(minDCFMVg)
    
    #modelPCADir(generativeModels.logMVG, 8) #calls logMVG with 8 directions and gives back it's minDCF
    
    minDFCValueslogMVG = []
    minDFCValueslogTiedMVG = []
    minDFCValueslogNaiveBayes = []
    minDFCValueslogTiedNaiveBayes = []
    i = 10
    #calling models with directions of 10(without PCA), 9, 8, 7, 6
    #while i > 5:
    #    minDFCValueslogMVG.append(modelPCADir(generativeModels.logMVG, i))
    #    minDFCValueslogTiedMVG.append(modelPCADir(generativeModels.logTiedMVG, i))
    #    minDFCValueslogNaiveBayes.append(modelPCADir(generativeModels.logNaiveBayes, i))
    #    minDFCValueslogTiedNaiveBayes.append(modelPCADir(generativeModels.logTiedNaiveBayes, i))
    #    i = i -1
    
    #Turning array of values to np arrays
    #array1 = np.array(minDFCValueslogMVG)
    #array2 = np.array(minDFCValueslogTiedMVG)
    #array3 = np.array(minDFCValueslogNaiveBayes)
    #array4 = np.array(minDFCValueslogTiedNaiveBayes)

    #concatanating arrays by lines
    #lines = []
    #lines.append("logMVG: " + ', '.join(map(str, array1)))
    #lines.append("logTiedMVG: " + ', '.join(map(str, array2)))
    #lines.append("logNaiveBayes: " + ', '.join(map(str, array3)))
    #lines.append("logTiedNaiveBayes: " + ', '.join(map(str, array4)))

    #result_array = np.array(lines)
    #print(result_array)
    #print(minDFCValues)
    
    file_path = 'result_array.npy'  # Path to save the numpy array

    #np.save(file_path, result_array) #saving numpy array as a file
    
    loaded_array = np.load(file_path) #loading numpy file to the program
    
    print(loaded_array)
    
    print("finished")