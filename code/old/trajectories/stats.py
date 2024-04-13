import numpy as np


def derivative1(vect:np.array)->np.array:
    """
    Function to compute the first derivative of some numpy array
    
    Args:
    -----
    - `vect`: numpy array of the N vectors of lenght T [NxT]
    
    Output:
    -------
    Numpy array with an estimation O(...) of the first derivative
    """
    
    if vect.ndim == 2:
        inds = np.arange(1,traj.shape[0])
        return vect[inds, :] - vect[inds-1, :]
    else:
        inds = np.arange(1,traj.shape[1])
        return vect[:, inds, :] - vect[:, inds-1, :]
    
    
def derivative2(vect:np.array)->np.array:
    """
    Function to compute the first derivative of some numpy array
    
    Args:
    -----
    - `vect`: numpy array of the N vectors of lenght T [NxT]
    
    Output:
    -------
    Numpy array with an estimation O(...) of the first derivative
    """
    
    pass


def getDistance(traj: np.array):
    """
    Gets the squared distances from the trajectory
    
    Args:
    -----
    - `traj`: np.array of N trajectories of length T [NxT]
    
    Output:
    -------
    np.array with the distances or corresponding float if only 1 trajecotry
    """
    
    if traj.ndim == 2:
        inds = np.arange(1,traj.shape[0])
        disp = (traj[inds, :] - traj[inds-1, :])**2
        
        return np.sum(dist, axis = 0)

    else:
        inds = np.arange(1,traj.shape[1])
        disp = (traj[:, inds, :] - traj[:, inds-1, :])**2
        
        return np.sum(dist, axis = 1)



def MSD(traj: np.array)-> np.array:
    """
    Allows to compute the Mean Squared Displacement of the trajectories for all timestamps
    
    Args:
    -----
    - `traj`: np.array of N trajectories of length T [NxT]
    
    Output:
    -------
    Mean Squared Displacement for all timestamps
    """
    
    if traj.ndim == 2:
        inds = np.arange(traj.shape[0])
        return (traj[inds,:] - traj[0, :])**2
        
    else:
        inds = np.arange(traj.shape[1])
        return (traj[:, inds, :] - traj[:, 0, :])**2
    
    
    

def getKurtosis():
    pass

def getSkewness():
    pass

def getMean():
    pass

def getSTD():
    pass

def getEntropy():
    pass

def getFractalDim():
    pass

def 