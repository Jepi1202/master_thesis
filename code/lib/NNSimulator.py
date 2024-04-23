import numpy as np
import torch
import yaml
import os
from tqdm import tqdm
from torch_geometric.data import Data
from norm import normalizeCol
from features import optimized_getGraph

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 


OUTPUT_TYPE = cfg['feature']['output']
THRESHOLD_DIST = cfg['feature']['distGraph']
MIN_DIST = cfg['normalization']['distance']['minDistance']
MAX_DIST = cfg['normalization']['distance']['maxDistance']
BOUNDARY = cfg['simulation']['parameters']['boundary']


MIN_X = cfg['normalization']['position']['minPos']
MAX_X = cfg['normalization']['position']['maxPos']
MIN_Y = cfg['normalization']['position']['minPos']
MAX_Y = cfg['normalization']['position']['maxPos']


NB_HIST = cfg['feature']['nbHist']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def distance(v1, v2):
    return np.linalg.norm(v1-v2)

###############
# Simulator
###############

class NNSimulator():
    def __init__(self, model):
        self.model = model


    def runSim(self, T, state, debug = None):
        return genSim(self.model, T, state, train = False, debug = debug)
        
        
def genSim(model, T, state, pos, train = True, debug = None):
    """ 
    Function to simulate froma model T timesteps

    Args:
    ----
        - `model`:
        - `T`:
        - `state`:
        - `pos`:
        - `train`:

    Ouptuts:
    --------
        historic of positions (list)
        and the list of speeds if training mode
    """

    pos = pos.to(DEVICE)
    hist = [torch.unsqueeze(pos, dim = 0)]          # [ [1, N, 2] ]

    # if training, keep the computation graph
    if train:

        yList = []  # list of speeds

        for i in range(T):
            # one-step transition

            if debug is not None:
                y = debug[i]
            else:
                y = model(state)
            yList.append(y)

            # update of the states and positions
            #with torch.no_grad():  # ?

            if OUTPUT_TYPE == 'speed':
                # Effect of boundary conditioned by previous position
                y = boundaryEffect(hist[-1][0], y, BOUNDARY )
                state, position = updateData(state, hist[-1][0], y)


            elif OUTPUT_TYPE == 'acceleration':
                v = state.x[:, :2]
                vNext = v + y
                vNext = boundaryEffect(hist[-1][0], vNext, BOUNDARY)
                state, position = updateData(state, hist[-1][0], vNext)

            hist.append(torch.unsqueeze(position, dim = 0))

        return torch.cat(hist, dim = 0), torch.cat(yList, dim = -1)       # [T, N, 2], [T, N, 2]

    # if not training, do not keep the comptutation graph
    else:
        model.eval()
        with torch.no_grad():

            for i in tqdm(range(T)):

                # one-step transition
                if debug is not None:
                    y = debug[i]
                else:
                    y = model(state)


                if OUTPUT_TYPE == 'speed':
                    # Effect of boundary conditioned by previous position
                    y = boundaryEffect(hist[-1][0], y, BOUNDARY )
                    state, position = updateData(state, hist[-1][0], y)

                elif OUTPUT_TYPE == 'acceleration':
                    v = state.x[:, :2]
                    vNext = v + y
                    vNext = boundaryEffect(hist[-1][0], vNext, BOUNDARY )               
                    state, position = updateData(state, hist[-1][0], vNext)


                hist.append(torch.unsqueeze(position, dim = 0))

        model.train()
        return torch.cat(hist, dim = 0)


def boundaryEffect(x:np.array, v:np.array, boundary:float = BOUNDARY):
    """ 
    Function to apply boundary effects

    Args:
    -----
        - `x`: positions
        - `v`: speeds
        - `boundary`: fronteers

    Returns:
    --------
        the speeds affected to 
    """
    
    for j in range(x.shape[0]):
        if x[j, 0] < - boundary or x[j, 0] > boundary:
            v[j, 0] = -v[j, 0]

        if x[j, 1] < - boundary or x[j, 1] > boundary:
            v[j, 1] = -v[j, 1]
        
    return v


def updateState(prevState, prevPos, speed):
    """ 
    Updates the prevStates into the following one

    Args:
    -----
        - `prevState`:previous state[[x, y, v_x, v_y]_t xN R]   [N, D]
        - `prevPos`: previous position
        - `speed`: speed [v_x, v_y]    [N, 2]
    """

    # get the next positions
    nextPos = prevPos + speed
        
    # update state
    R = prevState[:, -1]    # radius [N]

    # get new state by concatenation
    nextState = torch.cat((speed, prevState), dim = -1)

    s = torch.cat((nextState[:, :-3], torch.unsqueeze(R, dim = 1)), dim = -1)

    return s, nextPos


def getGraph(mat_t, threshold = THRESHOLD_DIST):
    """ 
    Function to compute the graph for pytorch geometric

    Args:
    -----
        - `mat_t` (np.array): 2D np array (mat at a given timestep)
        - `threshold` (float): 

    Returns:
    --------
        the list of indices and of distances for the graph
    """

    distList = []
    indices = []

    for i in range(mat_t.shape[0]):
        for j in range(i+1, mat_t.shape[0]):

            # compute the distance between cell at given timestep
            dist = distance(mat_t[i, :], mat_t[j, :])
            
            
            if dist < threshold:
                
                indices += [[i, j], [j, i]]

                direction_vector = mat_t[j, :] - mat_t[i, :]
                angle = np.arctan2(direction_vector[1], direction_vector[0])
                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)

                
                dist = normalizeCol(dist, MIN_DIST, MAX_DIST)
                
                distList.append(torch.tensor([dist, cos_theta, sin_theta], dtype=torch.float).unsqueeze(0))
                distList.append(torch.tensor([dist, -cos_theta, -sin_theta], dtype=torch.float).unsqueeze(0))


    indices = torch.tensor(indices)
    indices = indices.t().to(torch.long).view(2, -1)
    distList = torch.cat(distList, dim = 0)

    return distList, indices


def updateData(prevState, prevPose, speed, device = DEVICE, threshold = THRESHOLD_DIST):

    newS, nextPose = updateState(prevState.x,prevPose, speed)

    newGraph, newInds = optimized_getGraph(nextPose.cpu().detach().numpy().copy())

    return Data(x = newS, edge_index = newInds, edge_attr = newGraph).to(device), nextPose
