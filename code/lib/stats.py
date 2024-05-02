import numpy as np
import torch
import matplotlib.pyplot as plt


def MSD_comp(traj, tau):
    T = traj.shape[0]
    i = np.arange(T - tau)
    j = i + tau

    return np.linalg.norm(traj[j, :, :] - traj[i, :, :], axis=-1)**2

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

    res = []
    T = traj.shape[0]
    
    for tau in range(1, T):
        val = np.mean(np.mean(MSD_comp(traj, tau), axis=0), axis=0)
        res.append(val)

    return res


def applyMSD(sims:list, dislpay:bool = True)->np.array:
    """ 
    Function to apply MSD to a group of simulations

    NOTE: test

    Args:
    -----
        - `sims` (list): list of simualtions

    Returns:
    --------
        np array [#Sim, T-1] of MSD computations
    """

    res = np.zeros(len(sims), sims[0].shape[0]-1)
    for i in len(sims):
        sim = sims[i]
        res[i, :] = np.array(MSD(sim))


    if dislpay:
        x = np.arange(sims[0].shape[0]-1)
        y = np.mean(res, axis = 0)
        std = np.std(res, axis = 0)
        plt.plot(x, y, color = 'blue', zorder = 1)
        plt.fill_between(x, y - std, y+std, color = 'red', alpha = 0.4, zorder = 2)
        

    return res


####################################################################

