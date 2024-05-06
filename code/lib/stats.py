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

    res = np.zeros((len(sims), sims[0].shape[0]-1))
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



def scaleMagnVel(sim:np.array, bins:np.array, bins2:np.array, display:bool = True)->np.array:
    """ 
    Fucntion to compute the velocity maginutude distr

    Args:
    -----
        - `sim`: simulatioon 
        - `bins`: bins array
        - `bins2`: component histogram bins

    Returns:
    --------
        the tuple of velocities magnitude and components
    """

    # get the velocits

    inds0 = np.arange(sim.shape[0]-1)
    inds = inds0 + 1

    speeds = sim[inds] - sim[inds0]
    vmagn = np.linalg.norm(speeds, axis=-1)
    vx = speeds[:, :, 0]
    vy = speeds[:, :, 1]

    # average speed according to the cells
    avgSpeed = np.mean(vmagn, axis = -1)


    # intitialize the three histograms

    magnDist = np.zeros(len(bins)-1)
    magnDistX = np.zeros(len(bins2)-1)
    magnDistY = np.zeros(len(bins2)-1)


    # loop over cells
    for i in range(speeds.shape[0]):
                
        vdist,_=np.histogram(vmagn[i, :]/avgSpeed[i],bins,density=True)
        magnDist += vdist

        vdistx,_=np.histogram(vx[i, :]/avgSpeed[i],bins2,density=True)
        magnDistX += vdistx


        vdisty,_=np.histogram(vy[i, :]/avgSpeed[i],bins2,density=True)
        magnDistY += vdisty
    
    magnDist = magnDist/speeds.shape[0]
    magnDistX = magnDistX/speeds.shape[0]
    magnDistY = magnDistY/speeds.shape[0]
    


    if display:
        fig=plt.figure()
        db=bins[1]-bins[0]
        plt.grid()
        plt.semilogy(bins[1:]-db/2,magnDist,'r.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(frac{v}{<v>})')
        plt.title('Scaled velocity magnitude distribution')

        fig=plt.figure()
        db=bins2[1]-bins2[0]
        plt.grid()
        plt.semilogy(bins2[1:]-db/2,magnDistX,'r.-',lw=2)
        plt.semilogy(bins2[1:]-db/2,magnDistY,'k.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('$P(fracv/<v>)$')
        plt.title('Scaled velocity component (x & y) distribution')


    return magnDist, magnDistX, magnDistY


#b1=np.linspace(0,2,100)
#b2=np.linspace(-2,2,100)
#_ = scaleMagnVel(sim.copy(), bins = b1, bins2 = b2)




####################################################################

