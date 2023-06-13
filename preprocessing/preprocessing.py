import scipy
import numpy as np
import helpers.helpers as helpers

def computePCA(trainData : np.ndarray, m : int) -> np.ndarray:
    """
    Computes the first m Principal Components of the trainData dataset
    
    Args:
        trainData (np.ndarray): dataset
        m (int): 
            first m principal directions. 
            'm' gives the dimensionality of the projections that you want to obtain, ordered

    Returns:
        np.ndarray: simplified dataset
    """
    # mu should be a 1D array, of the means of the attributes
    mu = trainData.mean(1)

    # center the data
    centered_data = trainData - helpers.vcol(mu)

    # compute 1/N * Dc * DcT
    c = np.dot(centered_data, centered_data.T)/centered_data.shape[1]

    # compute eigenvalues (s) and eigenvectors (U) from small(!!) to big
    s, U = np.linalg.eigh(c)

    P = U[:, ::-1][:, 0:m]
    
    data_points = np.dot(P.T, trainData)
    
    print("explained_variance_ratio: ") 
    print(s[::-1][:m] / np.sum(s)) 
    print(s[::-1][:m]) 

    # Compute explained variance ratio 
    explained_variance_ratio = s[::-1][:m] / np.sum(s) 

    # Compute cumulative sum of explained variance ratios 
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio) 

    return data_points, cumulative_variance_ratio


def computeCumVarRatios(DTR):
    # Initialize an empty array to store cumulative variance ratios 
    cumulative_variance_ratios = [] 

    # Loop through different numbers of principal components 
    for m in range(0, DTR.shape[0] + 1): 
        # Compute PCA and get the cumulative variance ratio 
        _, cumulative_variance_ratio = computePCA(DTR, m) 
    
        print(f"Number of Principal Components: {m}, Cumulative Variance Ratio: {cumulative_variance_ratio} ,") 
    
        # Append the cumulative variance ratio to the array 
        if m == 0: 
            cumulative_variance_ratios.append(0) 
        else: 
            cumulative_variance_ratios.append(cumulative_variance_ratio[-1])  
            
    return cumulative_variance_ratios

def computeLDA(dataset: np.ndarray, labels : np.ndarray, m : int) -> np.ndarray:
    """
    Implements Linear Discriminant Analysis on 'dataset'
    Args:
        dataset (np.ndarray): dataset on which to implement the LDA
        labels (np.ndarray): labels of the dataset
        m (int): first m principal directions to keep

    Returns:
        np.ndarray: 
    """
    dataset_mean = dataset.mean(1)
    numclasses = max(labels) - min(labels) + 1
    nSamples = dataset.shape[1]
    
    S_b = np.zeros((dataset.shape[0], dataset.shape[0]), dtype = float)
    S_w = np.zeros((dataset.shape[0], dataset.shape[0]), dtype = float)
    
    for i in range(numclasses):
        class_samples = dataset[:, labels == i]
        class_mean = class_samples.mean(1)

        #S_b += class_samples.shape[1]*(class_mean - dataset_mean)*(class_mean - dataset_mean).T
        S_b += class_samples.shape[1]*(class_mean - dataset_mean)*(helpers.vrow((class_mean - dataset_mean)).T)

        centered_class_data = class_samples - helpers.vcol(class_mean)
        c_w_class = np.dot(centered_class_data, centered_class_data.T)/centered_class_data.shape[1]

        S_w += c_w_class*class_samples.shape[1]

    S_b /= nSamples
    S_w /= nSamples
    
    _, U = scipy.linalg.eigh(S_b, S_w)
    W = U[:, ::-1][:, 0:m]

    # orthogonalize W
    # Uw, _, _ = np.linalg.svd(W)
    # U = Uw[:, 0:m]

    reducedData = np.dot(W.T, dataset)
    return reducedData