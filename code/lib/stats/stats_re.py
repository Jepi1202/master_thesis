import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def MSD_comp(traj, tau):
    T = traj.shape[0]
    i = np.arange(T - tau)
    j = i + tau

    return (np.linalg.norm(traj[j, :, :] - traj[i, :, :], axis = -1))**2


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

def MSD_stat(sims:list, display:bool = False, color:str = 'blue', label = 'v0')->np.array:
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
    for i in range(len(sims)):
        sim = sims[i]
        res[i, :] = np.array(MSD(sim))


    if display:
        x = np.arange(sims[0].shape[0]-1)
        y = np.mean(res, axis = 0)
        std = np.std(res, axis = 0)
        plt.loglog(x, y, color = color, zorder = 1, label = label)
        plt.fill_between((x), (y - std), (y+std), color = color, alpha = 0.4, zorder = 2)
        
        plt.xlabel('Timesteps')
        plt.ylabel('MSD')
        plt.grid()

    return res








#######################################





def rdf(sim, boundary=120, nb_bins=720, max_dist=240):
    """ 
    sim [T, N, 2]
    """
    T, N, _ = sim.shape

    r = np.linspace(0.001, max_dist, nb_bins)
    dr = r[1] - r[0]
    g_r = np.zeros_like(r)

    for t in tqdm(range(T)):
        for i in range(N):
            # Vectorized computation of distances
            dist_vec = sim[t, i] - sim[t, i+1:N]
            dist = np.linalg.norm(dist_vec, axis=-1)

            # Filter distances that are within the max_dist
            valid_distances = dist[dist < max_dist]
            
            # Convert distances to bin indices
            indices = (valid_distances / dr).astype(int)
            
            # Increment g_r based on indices
            np.add.at(g_r, indices, 2)

    # Density of particles
    rho = N / (2 * boundary) ** 2

    # Normalization of g_r
    normalization_factor = 2 * np.pi * r * dr * rho * T * N
    g_r /= normalization_factor

    return g_r, r


def apply_rdf(simList):
    """ 
    [S, T, N, 2]
    """

    res = None
    for i in range(len(simList)):

        g_r, r = rdf(simList[i], boundary = 120, nb_bins = 50, max_dist = 240)

        if res is None:
            res = g_r

        else:
            res = np.vstack((res, g_r))


    mean_res = np.mean(res, axis = 0)
    std_res = np.std(res, axis = 0)

    return mean_res, std_res, r




#######################################


def SelfIntermediateA(data, qval, verbose=False):
    """ 
    [T, N, 2]
    """
    T, N, _ = data.shape  # T is the number of time steps, N is the number of particles
    qval = np.array(qval, dtype=np.complex128)  # Ensure wave vector is complex for the computation
    
    SelfInt = np.empty((T-1,), dtype=np.complex128)
    
    for t in range(T-1):
        smax = T - t  
        
        rt = data[:smax]       
        rtplus = data[t:] 
        dr = rt - rtplus
        
        exp_factor = np.exp(1j * (qval[0] * dr[:, :, 0] + qval[1] * dr[:, :, 1]))
        
        SelfInt[t] = np.sum(exp_factor) / (N * smax)
    
    SelfInt_mod = np.sqrt(np.real(SelfInt)**2 + np.imag(SelfInt)**2)
    
    tval = np.linspace(0, T-1, num=T-1)
    
    if verbose:
        plt.figure(figsize=(10, 5))
        plt.semilogy(tval, SelfInt_mod, '.-r', lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)
        plt.ylim([0, 1])
        plt.show()
    
    return tval, SelfInt_mod, SelfInt


def applySelfScattering(simList, qval = None, display:bool = False, color = 'blue'):
    """ 
    [S, T, N, 2]
    """
    if qval is None:
        R = 1       #ngfjdngjdnsjhnjfnvjfdnbjvnfdjkgnfjkdhvbjkdhgjklhgjkd
        qval = 2*np.pi/R*np.array([1,0])

    res = np.zeros((len(simList),simList[0].shape[0]-1 ))

    for i in range(len(simList)):
        sim = simList[i]

        val = SelfIntermediateA(sim.copy(), qval.copy(), verbose = False)[1]
        res[i] = val

    r = np.mean(res, axis=0)
    delta = np.std(r, axis = 0)

    if display:
        #plt.figure(figsize=(10, 5))
        t = np.arange(1, simList[0].shape[0])

        plt.grid()
        plt.semilogy(t, r, color = color, lw=2)
       # plt.fill_between(t, r-delta, r+delta, color = color, alpha = 0.4)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)

    return res



#############################################











#######################################


def SelfIntermediateA(data, qval, verbose=False):
    """ 
    [T, N, 2]
    """
    T, N, _ = data.shape  # T is the number of time steps, N is the number of particles
    qval = np.array(qval, dtype=np.complex128)  # Ensure wave vector is complex for the computation
    
    SelfInt = np.empty((T-1,), dtype=np.complex128)
    
    for t in range(T-1):
        smax = T - t  
        
        rt = data[:smax]       
        rtplus = data[t:] 
        dr = rt - rtplus
        
        exp_factor = np.exp(1j * (qval[0] * dr[:, :, 0] + qval[1] * dr[:, :, 1]))
        
        SelfInt[t] = np.sum(exp_factor) / (N * smax)
    
    SelfInt_mod = np.sqrt(np.real(SelfInt)**2 + np.imag(SelfInt)**2)
    
    tval = np.linspace(0, T-1, num=T-1)
    
    if verbose:
        plt.figure(figsize=(10, 5))
        plt.semilogy(tval, SelfInt_mod, '.-r', lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)
        plt.ylim([0, 1])
        plt.show()
    
    return tval, SelfInt_mod, SelfInt


def applySelfScattering(simList, qval = None, display:bool = False, color = 'blue'):
    """ 
    [S, T, N, 2]
    """
    if qval is None:
        R = 1       #ngfjdngjdnsjhnjfnvjfdnbjvnfdjkgnfjkdhvbjkdhgjklhgjkd
        qval = 2*np.pi/R*np.array([1,0])

    res = np.zeros((len(simList),simList[0].shape[0]-1 ))

    for i in range(len(simList)):
        sim = simList[i]

        val = SelfIntermediateA(sim.copy(), qval.copy(), verbose = False)[1]
        res[i] = val

    r = np.mean(res, axis=0)
    delta = np.std(r, axis = 0)

    if display:
        #plt.figure(figsize=(10, 5))
        t = np.arange(1, simList[0].shape[0])

        plt.grid()
        plt.semilogy(t, r, color = color, lw=2)
       # plt.fill_between(t, r-delta, r+delta, color = color, alpha = 0.4)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)

    return res



#############################################





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



def apply_hist(data, bins1, bins2):

    res_norm = None
    res_x = None
    res_y = None
    for i in range(len(data)):
        norm, x, y = scaleMagnVel(data[i], bins1, bins2, display = False)

        if res_norm is None:
            res_norm = norm
        else:
            res_norm = np.vstack((res_norm, norm))

        if res_x is None:
            res_x = x
        else:
            res_x = np.vstack((res_x, x))

        if res_y is None:
            res_y = y
        else:
            res_y = np.vstack((res_y, y))

    return res_norm, res_x, res_y






#################################

def sumSpeeds(data):
    """
    [S, T, N, 2]
    """

    inds1 = np.arange(data.shape[1]-1)
    inds2 = inds1 + 1

    speeds = np.sum(data[:, inds2] - data[:, inds1], axis = -2)

    return speeds







###################################




def apply_mse_roll(preds, gts):
    """ 
    [S, T, N, 2]
    """

    vals = (preds - gts) ** 2

    vals = np.mean(vals, axis = -1)

    vals = np.mean(vals, axis = -1)

    return vals         # [S, T]





