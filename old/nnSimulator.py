import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch_geometric.data import Data



with open('cfg.yml', 'r') as file:
    cfg = yaml.safe_load(file) 



OUTPUT_TYPE = cfg['feature']['output']
THRESHOLD_DIST = cfg['feature']['distGraph']
MIN_DIST = cfg['normalization']['distance']['minDistance']
MAX_DIST = cfg['normalization']['distance']['maxDistance']
BOUNDARY = cfg['simulation']['parameters']['boundary']


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def distance(v1, v2):
    return np.linalg.norm(v1-v2)

###############
# Normalization
###############


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




class NNSimulator():
    def __init__(self, model):

        self.model = model


    def runSim(self, T, state):

        return genSim(self.model, T, state, train = False)
        
        
def genSim(model, T, state, train = True):

    if train:
        hist = [torch.unsqueeze(state.x[:, :2], dim = 0)]
        yList = []

        for i in tqdm(range(T)):
            y = model(state)
            yList.append(y)

            if OUTPUT_TYPE == 'speed':
                y = boundaryEffect(state.x[:, :2], y, 40 )
                state = updateData(state, y)

            elif OUTPUT_TYPE == 'acceleration':
                pass

            hist.append(torch.unsqueeze(state.x[:, :2], dim = 0))

        return torch.cat(hist, dim = 0), torch.cat(y, dim = 0)
    
    else:
        with torch.no_grad():
            hist = [torch.unsqueeze(state.x[:, :2], dim = 0)]

            for i in tqdm(range(T)):
                y = model(state)

                if OUTPUT_TYPE == 'speed':
                    y = boundaryEffect(state.x[:, :2], y, 40 )
                    state = updateData(state, y)

                elif OUTPUT_TYPE == 'acceleration':
                    pass

                hist.append(torch.unsqueeze(state.x[:, :2], dim = 0))

            return torch.cat(hist, dim = 0)


def boundaryEffect(x, v, boundary = BOUNDARY):
    
    for j in range(x.shape[0]):
        if x[j, 0] < - boundary or x[j, 0] > boundary:
            v[j, 0] = -v[j, 0]

        if x[j, 1] < - boundary or x[j, 1] > boundary:
            v[j, 1] = -v[j, 1]
        
    return v


def updateState(prevState, speed):
    """ 
    Updates the prevStates into the following one

    Args:
    -----
        - `prevState`:previous state[[x, y, v_x, v_y]_t xN R]   [N, D]
        - `speed`: speed [v_x, v_y]    [N, 2]
    """

    # get the next positions
    nextPos = prevState[:, :2] + speed
    
    # update state
    d = prevState.shape[-1]
    R = prevState[:, -1]

    newS = torch.cat((nextPos, speed), dim = -1)
    nextState = torch.cat((newS, prevState), dim = -1)

    s = torch.cat((nextState[:, :-5], torch.unsqueeze(R, dim = 1)), dim = -1)

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





def updateData(prevState, speed, device = DEVICE, threshold = THRESHOLD_DIST):

    newS, nextPose = updateState(prevState.x, speed)

    newGraph, newInds = getGraph(nextPose.cpu().detach().numpy())

    return Data(x = newS, edge_index = newInds, edge_attr = newGraph).to(device)
