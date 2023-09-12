import numpy as np
import random

def loadData(filename : str, scramble : bool = False) -> tuple():
    """loads data from a txt file into a train and a test numpy array.

    Args:
        filename (string): the file that contains the dataset.
        scramble (bool, optional) Defaults to False. If yes, the data is going to be scrambled.
    Returns:
        Tuple(np.ndarray, np.ndarray): 
            returns two np arrays, one containing the samples of the data,
            the other the corresponding labels.
    """
    random.seed(0)
    data, labels = [], []
    
    lines = open(filename).readlines()
    
    if(scramble == True):
        random.shuffle(lines)
        

    for line in lines:
        linearray = line.rstrip().split(',')
        linelength = len(linearray)
        
        data.append([float(i) for i in linearray[0: linelength - 1]])
        labels.append(int(linearray[-1]))

    npdata = np.asarray(data).T
    nplabels = np.asarray(labels)
    
    return npdata, nplabels

