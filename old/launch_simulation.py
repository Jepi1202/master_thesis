import simulation as sim
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import os
import torch
from torch_geometric.data import Dataset, Data
import argparse

import features as ft

# normalization of radius of cells
MIN_RAD = 0.5
MAX_RAD = 30


#TODO ? make it automatic ?

def retrieveArgs():
    """ 
    Function to retrieve the args sent to the python code
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, default= None,
                        help="Model name used for saving the model after the training")
    parser.add_argument("--iter", type=int, default = 200,
                        help="Number of iterations to perform in the learning")
    parser.add_argument("--l1", type=int, default= 0,
                        help="Bool to force the l1 regularisation")
    parser.add_argument("--loss", type=str, default= 'mse',
                        help="Bool to force the l1 regularisation")
    parser.add_argument("--wb", type=str, default= 'new-run',
                        help="Name of the wandb run")
    

    args = vars(parser.parse_args())

    return args




def normalizeCol(vect: np.array, minVal:float, maxVal:float)->np.array:
    """ 
    Function used in order to apply min-max stand on features

    Args:
    -----
        -`vect` (np.array): array to normalize
        - `minVal` (float): min value in min-max stand
        - `maxVal` (float): max value in min-max stand

    Returns:
    --------
        the normalized vector
    """
    assert minVal < maxVal
    
    ran = maxVal - minVal
    return (vect - minVal)/ran



class SimulationParameters():
    def __init__(self):
        
        # Default values
        self.n = 20      # number of cells
        self.v0 = 50     #
        self.tau = 0.5
        self.k = 10
        self.epsilon = 0.3
        
        self.boundary = 50
        
        self.T = 100
        self.dt = 0.01
        
        self.seed = np.random.randint(0, 100000000)
        
        self.cutoff = 10           # use it in the features ??
        self.radii = 1.0
        self.noiseBool = False
        
        self.p = (self.v0, self.tau, self.k, self.epsilon)
        
    def _getParams(self):
        paramList = np.array([normalizeCol(self.radii, MIN_RAD, MAX_RAD)])
        return paramList
    
    def _loadParams(self, vect):
        self.n = vect[0]      # number of cells
        self.v0 = vect[1]     #
        self.tau = vect[2]
        self.k = vect[3]
        self.epsilon = vect[4]
        
        self.boundary = vect[5]
        
        self.T = vect[6]
        self.dt = vect[7]
        
        self.seed = vect[8]
        
        self.cutoff = vect[9]           # use it in the features ??
        self.radii = vect[10]
        self.noiseBool = vect[11]
        
        self.p = (self.v0, self.tau, self.k, self.epsilon)
        
    def _getSimParams(self):
        return self.n.astype(np.int32), self.p, self.boundary, self.T.astype(np.int32), self.dt, self.seed, self.cutoff, self.radii, self.noiseBool


def run_sim(path,pathSim,  simNb, n, v0, k, boundary, epsilon, tau, T, dt, seeds, radii, cutoffs,nbHist, noiseBool, saveBool:bool = False,saveSimBool:bool =False, completePath:bool = False):

    params = SimulationParameters()


    n_i = n.astype(np.int32)
    v0_i = v0
    k_i = k
    boundary_i = boundary
    epsilon_i = epsilon
    tau_i = tau
    T_i = T
    dt_i = dt

    seed_i = seeds
    
    radii_i = radii
    cutoff = cutoffs

    params._loadParams([n_i, v0_i, tau_i, k_i, epsilon_i, boundary_i, T_i, dt_i, seed_i, cutoff, radii_i, noiseBool])
    
    p = params._getSimParams()

    #print(params)
    resOutput, resInter = sim.compute_main(*p)  
    paramList = params._getParams()

    if saveSimBool:
        # saving np file for the simulation dataloader
        data_dict = {'resOutput': resOutput, 'paramList': paramList}
        np.save(os.path.join(pathSim, f'output_{simNb}'), data_dict)
    
    nodesFeatures, yVect = ft.getFeatures(resOutput, paramList, nbHist)
    edgeIndexVect, edgeFeaturesVect = ft.getEdges(resOutput, params.cutoff)


    if saveBool:
        idx = 0
        for j in range(len(nodesFeatures)):

            x = nodesFeatures[j]
            y = yVect[j]

            edgeFeature = edgeFeaturesVect[j]
            edgeIndex = edgeIndexVect[j]


            data = Data(x=x, 
                        edge_index=edgeIndex,
                        edge_attr=edgeFeature,
                        y=y
                        ) 

            if completePath:
                torch.save(data, path)
            else:

                torch.save(data, os.path.join(path, f'data_sim_{simNb}_nb_{idx}_v2_basic.pt'))
            idx += 1


    return nodesFeatures, yVect, edgeIndexVect, edgeFeaturesVect


        
def create_data(path, pathSim, nbSim, n, v0, k, boundary, epsilon, tau, T, dt, seeds, radii, cutoffs,nbHist, noiseBool, saveBool:bool = False, saveSimBool:bool = False, completePath:bool = False):
    
    for i in tqdm(range(len(seeds))):

        n_i = n[i].astype(np.int32)
        v0_i = v0[i]
        k_i = k[i]
        boundary_i = boundary[i]
        epsilon_i = epsilon[i]
        tau_i = tau[i]
        T_i = T[i]
        dt_i = dt[i]

        seed_i = seeds[i]
        
        radii_i = radii[i]
        cutoff = cutoffs[i]


        if completePath:
            _ = run_sim(path[i], pathSim[i],  i, n_i, v0_i, k_i, boundary_i, epsilon_i, tau_i, T_i, dt_i, seed_i, radii_i, cutoff, nbHist, noiseBool, saveBool,saveSimBool = saveSimBool, completePath = completePath)
        else:
            _ = run_sim(path, pathSim,  i, n_i, v0_i, k_i, boundary_i, epsilon_i, tau_i, T_i, dt_i, seed_i, radii_i, cutoff, nbHist, noiseBool, saveBool)



def generate_conds(nbSim):

    n = np.random.uniform(20, 120, nbSim).astype(int)
    v0 = np.ones(nbSim) * 40
    k  = np.ones(nbSim) * 10
    boundary = np.ones(nbSim) * 75
    epsilon = np.ones(nbSim) * 0.3
    tau = np.ones(nbSim) * 0.5


    T = np.ones(nbSim) * 100
    dt = np.ones(nbSim) * 0.01

    seeds = np.random.randint(0, 100000000, nbSim)

    thresholdDist = np.ones(nbSim) *10
    R = np.ones(nbSim) *10.0



    return n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R





def full_gen(nbSim, path, pathSim, nbHist:int = 2, noiseBool:bool = False, saveBool: bool = True,saveSimBool: bool = False, completePath:bool = True):

    n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R = generate_conds(nbSim)

    for i in tqdm(range(nbSim)):
        _ = run_sim(path[i], pathSim[i],  i, n, v0, k, boundary, epsilon, tau, T, dt, seeds, R, thresholdDist, nbHist, noiseBool, saveBool,saveSimBool = saveSimBool, completePath = completePath)



