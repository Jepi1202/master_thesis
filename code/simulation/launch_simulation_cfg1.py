import numpy as np
import torch
from torch_geometric.data import Data
import yaml
from tqdm import tqdm
import os
import sys

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

import features as ft
import simulation_v2 as sim

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'data_creation_cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file)['cfg1'] 


SIM = cfg['simulation']
PARAMS = SIM['parameters']
# displayig simulation parameters
print(PARAMS)


# number of runs:
NB_LEARNING = SIM['nbSimLearning']
NB_VAL = SIM['nbValidation']
NB_TEST = SIM['nbTest']
INIT_SIM = SIM['initialization']        # kind of initilization
INIT_DIST = SIM['initDistance']         # dsitance between nodes at the beginning
INIT_NB = SIM['initNb']                 # number of nodes - 1
INIT_NB_POSE = SIM['initNbPos']
INIT_TOT = INIT_NB_POSE * INIT_NB_POSE * (INIT_NB + 1)

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
#NB_HIST = cfg['feature']['nbHist']
NB_HIST = 4

#############################################################

SCRATCH = cfg['path']['simulationPath']
p = SCRATCH

print(f'Deposing files in depository >>>> {p}')
print(f'Number of learning simulations >>> {NB_LEARNING}')
print(f'Number of validation simulations >>> {NB_VAL}')
print(f'Number of test simulations >>> {NB_TEST}')
print(f'Number of lagged values >>> {NB_HIST}')

# shift the files names
TORCH_MIN = 0       # change if want to detect already existing files  


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
        
    def _getParams(self)->np.array:
        #paramList = np.array([self.radii])
        #return paramList
        return None
    
    def _loadParams(self, vect, force:bool = False)->None:
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
        
    def _getSimParams(self)->tuple:
        """ 
        Returns the parameters in the good format for the simulation
        """
        return self.n.astype(np.int32), self.p, self.boundary, self.T.astype(np.int32), self.dt, self.seed, self.cutoff, self.radii, self.noiseBool
    

    def _getInfo(self)->dict:
        """ 
        Constructs a dictionnary with informations on the simulation within
        """
        d = {}

        d['n'] = self.n
        d['v0'] = self.v0
        d['tau'] = self.tau
        d['k'] = self.k
        d['epsilon'] = self.epsilon
        d['boundary'] = self.boundary
        d['T'] = self.T
        d['dt'] = self.dt
        d['seed'] = self.seed
        d['cutoff'] = self.cutoff
        d['radii'] = self.radii
        d['noiseBool'] = self.noiseBool

        return d


############################################


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
        
        if not os.path.exists(p):
            os.makedirs(p)
            os.makedirs(p_np)
            os.makedirs(p_torch)
        
        res.append((p, p_np, p_torch))
                   
    return res      # [[path, path_numpy, path_torch]]


############################################

def initialConditions(n:int = None)->tuple:
    """ 
    Generate intial conditions for the simulations
    Gives the initial positions and the initial angles ([0, 2 pi])
    of the cells 

    Args:
    -----
        - `n` (optional): number of cells 

    Returns:
    --------
        tuple (pos, angles)
        pos: np.array [N, 2]
        angles: np.array [N]
    """

    if INIT_SIM == 'easy':
        lim = 0.85 * BOUNDARY
        deltaDist = INIT_DIST

        nb = INIT_NB

        xPos = np.linspace(-lim, lim, INIT_NB_POSE)
        yPos = np.linspace(-lim, lim, INIT_NB_POSE)
        gridX, gridY = np.meshgrid(xPos, yPos)
        pos = np.stack([gridX.ravel(), gridY.ravel()], axis=1)
        
        for i in range(nb):
            delta = np.random.uniform(0, deltaDist, gridX.shape + (2,))

            gridX2 = gridX + delta[:, :, 0]
            gridY2 = gridY + delta[:, :, 1]

            pos_perturbed = np.stack([gridX2.ravel(), gridY2.ravel()], axis=1)

            pos = np.concatenate([pos, pos_perturbed], axis=0)

        angles = np.random.rand(pos.shape[0]) * 2 * np.pi

        return (pos, angles)
    

    elif INIT_SIM == 'circle':
        return None
    

    elif INIT_SIM == 'random':
        assert n is not None

        lim = 0.85 * BOUNDARY
        pos = np.random.uniform(-lim , lim ,(n,2))

        angles = np.random.rand(n) * 2 * np.pi

        return (pos, angles)


 ############################################       

def generateParams(nbSim:int)->tuple:
    """
    Generates the parameters for the simulator

    Args:
    -----
        - `nbSim` (int): number of simulations

    Returns:
    --------
        the parameters of the simlation
        (n, v0, k, boundary, epsilon, tau, T, dt, seeds, thresholdDist, R)
        - n: number of cells
        - v0: active speed
        - k
        - boundary
        - epsilon
        - tau
        - T
        - dt
        - seeds
        - thresholdDist
        - R
    """

    # seed
    seeds = np.random.randint(0, 100000000, nbSim)

    # simulation parameters
    n = np.random.uniform(NB_CELLS_MIN, NB_CELLS_MAX, nbSim).astype(int)    # 60%
    if INIT_SIM == 'easy':
        n = np.ones(nbSim) * INIT_TOT
    v0 = np.ones(nbSim) * VO_PARAMS
    k  = np.ones(nbSim) * K_PARAMS
    boundary = np.ones(nbSim) * BOUNDARY
    epsilon = np.ones(nbSim) * EPSILON
    tau = np.ones(nbSim) * TAU
    T = np.ones(nbSim) * T_PARAM
    dt = np.ones(nbSim) * DT_PARAM
    R = np.ones(nbSim) *R_PARAM

    # graph construction
    thresholdDist = np.ones(nbSim) *THRESHOLD_DIST


    res = []
    d = {}

    for i in range(nbSim):
        parameters = SimulationParameters()
        parameters._loadParams([n[i], v0[i], tau[i], k[i], epsilon[i], boundary[i], T[i], dt[i], seeds[i], thresholdDist[i], R[i], NOISY_BOOL], force = True)
        res.append(parameters)
        d[i] = parameters._getInfo()

    return res, d


############################################


def runSim(pathTorch:str, pathSim:str, params:SimulationParameters, cond:tuple, nbHist:int = NB_HIST, saveBool:bool = False)-> tuple:
    """ 
    Function to run the simulation

    Args:
    -----
        - `pathTorch`: path to the torch files
        - `pathSim`: path to the simulations
        - `params`: parameters of the simulation
        - `cond`: tuple of conditions
        - `nbHist`: number of lagged values
        - `saveBool`: boolean for saving or not

    Retunrs:
    --------

    """
    
    p = params._getSimParams()

    # run the simulation

    resOutput, resInter = sim.compute_main(*p, initialization = cond)  

    # save simulation
    np.save(pathSim, resOutput)

    # create the features of the simulation
    nodesFeatures, yVect, edgeFeaturesVect, edgeIndexVect = ft.processSimulation(resOutput, nbhist = nbHist)

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

            #torch.save(data, os.path.join(pathTorch, f'data_sim_{simNb + TORCH_MIN}_nb_{idx}_v2_basic.pt'))
            torch.save(data, f'{pathTorch}_step_{idx}.pt')
            idx += 1


    return nodesFeatures, yVect, edgeIndexVect, edgeFeaturesVect


############################################


def create_data(nbSim:int, pathTorch:str, pathSim:str)->None:

    """ 
    Function for creating the 

    Args:
    -----
        - ``

    Returns:
    --------
        - ``
    """

    # generate conditions
    params, infos = generateParams(nbSim)


    # create simulations
    for i in tqdm(range(len(params))):

        # loop throught the parameters
        n_i = params[i].n.astype(np.int32)
        
        # get initial conditions
        conds = initialConditions(n_i)

        # create the path for torch and np files
        pathT = os.path.join(pathTorch, f'sim_{i}')
        pathS = os.path.join(pathSim, f'simulation_{i}.npy')

        # create simulation
        _ = runSim(pathT, pathS,  params[i], conds, saveBool=True)

    # save dict
    #TODO


############################################

def main():

    #NOTE: could be more concise but remains as such for better maniability
    nbSimLearning = NB_LEARNING
    nbSimVal = NB_VAL
    nbSimTest = NB_TEST

           
    paths = generateFolder(p)
    nbHist =  NB_HIST

    #########################################
    # LEARNING
    #########################################


    path, npPath, torchPath = paths[0]

    _ = create_data(nbSimLearning, torchPath, npPath)
    
    #########################################
    # validation
    #########################################
    
    path, npPath, torchPath = paths[1]
    
    _ = create_data(nbSimVal, torchPath, npPath)    

    #########################################
    # test
    #########################################
    
    path, npPath, torchPath = paths[2]
    
    _ = create_data(nbSimTest, torchPath, npPath)  

        
if __name__ == '__main__':

    main()
    