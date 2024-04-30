import numpy as np
import torch
import yaml
import os
from tqdm import tqdm
from torch_geometric.data import Data
from norm import normalizeCol
from features import optimized_getGraph, getFeatures

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


R_PARAM = cfg['simulation']['parameters']['R']
MIN_RAD = cfg['normalization']['radius']['minRad']
MAX_RAD = cfg['normalization']['radius']['maxRad']

MIN_DELTA = -8
MAX_DELTA = 8


NB_HIST = cfg['feature']['nbHist']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###############
# Simulator
###############

class NNSimulator():
    def __init__(self, model):
        self.model = model


    def runSim(self, T, state, pos, debug = None):
        return genSim(self.model, T, state, pos, train = False, debug = debug)
        
        
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

            # update of the states and positions
            #with torch.no_grad():  # ?

            if OUTPUT_TYPE == 'speed':
                # Effect of boundary conditioned by previous position
                nPose = hist[-1][0] + y
                vNext = boundaryEffect(nPose, y, BOUNDARY )
                
            elif OUTPUT_TYPE == 'acceleration':
                v = state.x[:, :2]
                vNext = v + y
                nPose = hist[-1][0] + vNext
                vNext = boundaryEffect(nPose, vNext, BOUNDARY)

            yList.append(vNext)
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
                    nPose = hist[-1][0] + y
                    vNext = boundaryEffect(nPose, y, BOUNDARY )

                elif OUTPUT_TYPE == 'acceleration':
                    v = state.x[:, :2]
                    vNext = v + y
                    nPose = hist[-1][0] + vNext
                    vNext = boundaryEffect(nPose, vNext, BOUNDARY )               

                state, position = updateData(state, hist[-1][0], vNext)
                hist.append(torch.unsqueeze(position, dim = 0))

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


def updateData(prevState, prevPose, speed, device = DEVICE, threshold = THRESHOLD_DIST):

    newS, nextPose = updateState(prevState.x,prevPose, speed)

    newGraph, newInds = optimized_getGraph(nextPose.cpu().detach().numpy().copy())
    
    newGraph = normalizeCol(newGraph, MIN_DELTA, MAX_DELTA)

    return Data(x = newS, edge_index = newInds, edge_attr = newGraph).to(device), nextPose


def getSimulationVideo(model:torch.tensor, initPos:torch.tensor, nbTimesteps:int, initState:torch.tensor) -> torch.tensor:
    """ 
    Function to create a simulation from the model

    Args:
    -----
        - `model`
        - `initPos`
        - `initState`
        - `nbTimesteps`
        - `outputPath`

    Retruns:
    --------
        tensor of the different positions
    """

    simulator = NNSimulator(model)

    res = simulator.runSim(nbTimesteps, initState, initPos)

    #create_simulation_video_cv2(res, outputPath, fps = 10, size = (600,600))

    return res



def getSimulationData(model:torch.tensor, nbTimesteps:int, d:np.array, i = 5, display =True) -> torch.tensor:
    x, y = getFeatures(d.copy(), np.array([R_PARAM]), nb = 4)
    attr, inds = optimized_getGraph(d[5, :, :].copy())
    attr = normalizeCol(attr, MIN_DELTA, MAX_DELTA)
    s = Data(x = x[4][: , 2:], edge_attr = attr, edge_index = inds).to(DEVICE)

    res = getSimulationVideo(model, torch.from_numpy(d[5, :, :].copy()).float(), nbTimesteps, s)
    
    return res