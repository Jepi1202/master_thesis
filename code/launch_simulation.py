import numpy as np
import torch
from torch_geometric.data import Data
import argparse
import yaml
from tqdm import tqdm
import os
import features as ft
import simulation as sim
from norm import normalizeCol


with open('cfg.yml', 'r') as file:
    cfg = yaml.safe_load(file) 
    
NORMALIZATION = cfg['normalization']
SIM = cfg['simulation']
PARAMS = SIM['parameters']
print(PARAMS)


# normalization of radius of cells
MIN_RAD = NORMALIZATION['radius']['minRad']
MAX_RAD = NORMALIZATION['radius']['maxRad']


# number of runs:
NB_LEARNING = SIM['nbSimLearning']
NB_VAL = SIM['nbValidation']
NB_TEST = SIM['nbTest']


# parameters of the simulation
NOISY_BOOL = PARAMS['noisy']
NB_CELLS_MIN = PARAMS['nMin']
NB_CELLS_MAX = PARAMS['nMax']
VO_PARAMS = PARAMS['v0']
K_PARAMS = PARAMS['k']
EPSILON = PARAMS['epsilon']
TAU = PARAMS['tau']
T_PARAM = PARAMS['T']
DT_PARAM = PARAMS['dt']
THRESHOLD_DIST = PARAMS['threshold']
R_PARAM = PARAMS['R']
BOUNDARY = PARAMS['boundary']


# nb hist:
NB_HIST = cfg['feature']['nbHist']



SCRATCH = '/scratch/users/jpierre/test_speed'
#p = '/home/jpierre/v2/part_1_b'
p = SCRATCH
print(f'Deposing files in depository >>>> {p}')
print(f'Number of learning simulations >>> {NB_LEARNING}')
print(f'Number of validation simulations >>> {NB_VAL}')
print(f'Number of test simulations >>> {NB_TEST}')
print(f'Number of lagged values >>> {NB_HIST}')

# parameters to fine-tune by hand in order to
# shift the files names

TORCH_MIN = 0       


#TODO Adapt the rest to automatically change the cfg file ....

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


class SimulationParameters():
    """ 
    Class to keep all the simulation parameters
    """
    def __init__(self):
        
        # Default values
        self.n = NB_CELLS_MIN     
        self.v0 = VO_PARAMS   
        self.tau = TAU
        self.k = K_PARAMS
        self.epsilon = EPSILON
        
        self.boundary = BOUNDARY
        
        self.T = T_PARAM
        self.dt = DT_PARAM
        
        self.seed = np.random.randint(0, 100000000)
        
        self.cutoff = THRESHOLD_DIST          
        self.radii = R_PARAM
        self.noiseBool = NOISY_BOOL
        
        self.p = (self.v0, self.tau, self.k, self.epsilon)
        
    def _getParams(self):
        paramList = np.array([normalizeCol(self.radii, MIN_RAD, MAX_RAD)])
        return paramList
    
    def _loadParams(self, vect, force:bool = False):
        self.n = vect[0] 
        if force:   
            self.v0 = vect[1]     
            self.tau = vect[2]
            self.k = vect[3]
            self.epsilon = vect[4]
            self.boundary = vect[5]
            self.T = vect[6]
            self.dt = vect[7]
            self.seed = vect[8]
            self.cutoff = vect[9]
            self.radii = vect[10]
            self.noiseBool = vect[11]
            
        self.p = (self.v0, self.tau, self.k, self.epsilon)
        
    def _getSimParams(self):
        """ 
        Returns the parameters in the good format for the simulation
        """
        return self.n.astype(np.int32), self.p, self.boundary, self.T.astype(np.int32), self.dt, self.seed, self.cutoff, self.radii, self.noiseBool


def run_sim(path,pathSim,  simNb, n, v0, k, boundary, epsilon, tau, T, dt, seeds, radii, cutoffs,nbHist, noiseBool, saveBool:bool = False,saveSimBool:bool =False, completePath:bool = False, force:bool = True):
    """ 
    Function to run the simulation

    Args:
    -----
        - ``

    Retunrs:
    --------

    """

    params = SimulationParameters()

    # load parameters
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

    params._loadParams([n_i, v0_i, tau_i, k_i, epsilon_i, boundary_i, T_i, dt_i, seed_i, cutoff, radii_i, noiseBool], force = force)
    
    p = params._getSimParams()

    # run the simulation
    resOutput, resInter = sim.compute_main(*p)  
    paramList = params._getParams()

    #TODO update here
    # save parameters
    data_dict = {'resOutput': resOutput, 'paramList': paramList}

    # save simulation
    np.save(os.path.join(pathSim, f'output_{simNb}'), data_dict)
    
    # create the features of the simulation
    nodesFeatures, yVect = ft.getFeatures(resOutput, paramList, nbHist)
    edgeIndexVect, edgeFeaturesVect = ft.getEdges(resOutput, params.cutoff)

    # save the pt files
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

                torch.save(data, os.path.join(path, f'data_sim_{simNb + TORCH_MIN}_nb_{idx}_v2_basic.pt'))
            idx += 1


    return nodesFeatures, yVect, edgeIndexVect, edgeFeaturesVect


def create_data(path, pathSim, nbSim, n, v0, k, boundary, epsilon, tau, T, dt, seeds, radii, cutoffs,nbHist, noiseBool, saveBool:bool = False, saveSimBool:bool = False, completePath:bool = False):
    """ 
    Calls in loop run_sim in order to save the simulations
    NOTE: just for conciseness of the code
    """

    for i in tqdm(range(len(seeds))):

        # loop throught the parameters
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


"""

OLD FUNCTION TO CREATE THE PARAMETERS
def generate_conds(nbSim):

    n = np.random.uniform(150, 300, nbSim).astype(int)    # 60%
    v0 = np.ones(nbSim) * 10    #reduce ~ 
    k  = np.ones(nbSim) * 10
    boundary = np.ones(nbSim) * 40
    epsilon = np.ones(nbSim) * 0.3
    tau = np.ones(nbSim) * 0.5


    T = np.ones(nbSim) * 400       # slow down, more timesteps
    dt = np.ones(nbSim) * 0.01

    seeds = np.random.randint(0, 100000000, nbSim)

    thresholdDist = np.ones(nbSim) *6.0
    R = np.ones(nbSim) *1.0



    return n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R
"""


def generate_conds(nbSim):

    n = np.random.uniform(NB_CELLS_MIN, NB_CELLS_MAX, nbSim).astype(int)    # 60%
    v0 = np.ones(nbSim) * VO_PARAMS    #reduce ~ 
    k  = np.ones(nbSim) * K_PARAMS
    boundary = np.ones(nbSim) * BOUNDARY
    epsilon = np.ones(nbSim) * EPSILON
    tau = np.ones(nbSim) * TAU


    T = np.ones(nbSim) * T_PARAM       # slow down, more timesteps
    dt = np.ones(nbSim) * DT_PARAM

    seeds = np.random.randint(0, 100000000, nbSim)

    thresholdDist = np.ones(nbSim) *THRESHOLD_DIST
    R = np.ones(nbSim) *R_PARAM



    return n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R

        
def generateFolder(path:str):
    """ 
    Function to create the folder where to put the data

    Args:
    -----
        - `path` (str): path where to load the data

    Returns:
    --------
        The list of the [paht, paht_numpy, path_torch] for the 
        training, validation and test
    """
    
    res = []
    
    p_train = os.path.join(path, 'training')
    p_val = os.path.join(path, 'validation')
    p_test = os.path.join(path, 'test')
    
    for p in [p_train, p_val, p_test]:
        p_np = os.path.join(p, 'np_file')
        p_torch = os.path.join(p, 'torch_file')
        
        os.makedirs(p)
        os.makedirs(p_np)
        os.makedirs(p_torch)
        
        res.append((p, p_np, p_torch))
                   
    return res      # [[path, path_numpy, path_torch]]
        

def main():

    #NOTE: could be more concise but remains as such for better maniability
           
    paths = generateFolder(p)
    nbHist =  NB_HIST

    #########################################
    # LEARNING
    #########################################

    nbSim = NB_LEARNING
    
    n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R = generate_conds(nbSim)
    
    path, npPath, torchPath = paths[0]

    _ = create_data(torchPath, npPath, nbSim, n, v0, k, boundary, epsilon, tau, T, dt, seeds, R, thresholdDist, nbHist, noiseBool = False, saveBool = True, saveSimBool = True, completePath = False)
    

    #########################################
    # validation
    #########################################


    nbSim = NB_VAL
    
    n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R = generate_conds(nbSim)
    
    path, npPath, torchPath = paths[1]
    
    _ = create_data(torchPath, npPath, nbSim, n, v0, k, boundary, epsilon, tau, T, dt, seeds, R, thresholdDist, nbHist, noiseBool = False, saveBool = True, saveSimBool = True, completePath = False)
    

    #########################################
    # test
    #########################################
    
    nbSim = NB_TEST
    
    n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R = generate_conds(nbSim)
    
    path, npPath, torchPath = paths[2]
    
    _ = create_data(torchPath, npPath, nbSim, n, v0, k, boundary, epsilon, tau, T, dt, seeds, R, thresholdDist, nbHist, noiseBool = False, saveBool = True, saveSimBool = True, completePath = False)


        
if __name__ == '__main__':

    main()
    