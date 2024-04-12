import numpy as np
import torch
from typing import Optional


def keepBounds(sim, bound):

    inds = np.where(sim < bound)




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



def applyMSD(sims:list)->np.array:
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

    return res




def speedVelocity_comp(sim, res):

    # ground truth
    i = np.arange(sim.shape[0]-1)       # [0, 98]
    j = i + 1                           # [1, 99]

    vel = sim[j, :, :] - sim[i, :, :]
    vel = np.linalg.norm(vel, axis=-1)      #[0->1, ..., 98->99]

    meanVelPred = np.mean(vel, axis = -1)

    # predictions
    velGT = []
    
    for i in range(len(res)):
        velGT.append(np.linalg.norm(res[i]))

    return vel


def infere(model, graph, t):

    x = graph.x
    pos = [x]
    for i in range(t):
        v = model(graph)

        x += v

        # get data graph from x





def applyDegrees(sim, cutoff):
    """ 
    Function to compute the degrees of each node

    #TODO: test that ...
    """
    N = sim.shape[1]    # number of nodes
    T = sim.shape[0]    # number of timesteps

    res = np.zeros((T, N))
    for i in range(T):
        for j in range(N):
            # put that in a function
            d = np.linalg.norm(sim[i, :, :] - sim[i, j, :], axis = -1)
            Nneighbors = len(np.where(d < cutoff)[0])
            res[i, j] = Nneighbors

    return res





def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"

        https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)


from sklearn import metrics

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]

    https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py 
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


##########################################
# Display functions
##########################################
