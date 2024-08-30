import numpy as np
import torch
import yaml
import os
from tqdm import tqdm
from torch_geometric.data import Data
from norm import normalizeGraph
from features import optimized_getGraph, getFeatures, processSimulation

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 


OUTPUT_TYPE = cfg['feature']['output']
THRESHOLD_DIST = cfg['feature']['distGraph']
MIN_DIST = cfg['normalization']['distance']['minDistance']
MAX_DIST = cfg['normalization']['distance']['maxDistance']
BOUNDARY = cfg['simulation']['parameters']['boundary']
R_PARAM = cfg['simulation']['parameters']['R']


NB_HIST = cfg['feature']['nbHist']

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###############
# Simulator
###############

class NNSimulator():
    def __init__(self, model):
        self.model = model


    def runSim(self, T, state, pos, debug = None, train = False):
        return genSim(self.model, T, state, pos, train = train, debug = debug)
        
        
def genSim(model, T, state, pos, radius=None, train = True, debug = None):
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
    
    #if radius is None:
    radius = np.ones(pos.shape[0]) * R_PARAM

    pos = pos.to(DEVICE)
    hist = [torch.unsqueeze(pos, dim = 0)]          # [ [1, N, 2] ]

    # if training, keep the computation graph
    if train:

        yList = []  # list of speeds

        for i in range(T):
            
            # one-step transition

            if debug is not None:
                y = model(debug[i]).to(DEVICE)
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
            state, position = updateData(state, hist[-1][0], vNext, radius = radius)
            hist.append(torch.unsqueeze(position, dim = 0))

        return torch.cat(hist, dim = 0), torch.stack(yList)       # [T, N, 2], [T, N, 2]

    # if not training, do not keep the comptutation graph
    else:
        model.eval()
        with torch.no_grad():

            for i in tqdm(range(T)):

                # one-step transition
                if debug is not None:
                    y = model(debug[i]).to(DEVICE)
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

                state, position = updateData(state, hist[-1][0], vNext, radius = radius)
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
        

    # get new state by concatenation
    nextState = torch.cat((speed, prevState[:, :-2]), dim = -1)

    return nextState, nextPos


def updateData(prevState, prevPose, speed, radius, device = DEVICE, threshold = THRESHOLD_DIST):

    newS, nextPose = updateState(prevState.x,prevPose, speed)

    newGraph, newInds = optimized_getGraph(nextPose.cpu().detach().numpy().copy(), radius = radius)

    data = Data(x = newS, edge_index = newInds, edge_attr = newGraph)
    
    data = normalizeGraph(data)

    return data.to(device), nextPose


def getSimulationVideo(model:torch.tensor, initPos:torch.tensor, nbTimesteps:int, initState:torch.tensor, train = False, debug = None) -> torch.tensor:
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

    res = simulator.runSim(nbTimesteps, initState, initPos, train = train, debug = debug)

    #create_simulation_video_cv2(res, outputPath, fps = 10, size = (600,600))

    return res


# only different function
def getSimulationData(model:torch.tensor, nbTimesteps:int, d:np.array, i = 5, display =True, train = False, debug = None, radius = None) -> torch.tensor:
    if radius is None:
        radius = np.ones(d.shape[1]) * R_PARAM

    x, y, attr, inds = processSimulation(d)
    data = Data(x = x[i][:, 2:], y = y[i] , edge_attr = attr[i], edge_index = inds[i])
    pos = x[i][:, :2]
    s = normalizeGraph(data).to(DEVICE)

    res = getSimulationVideo(model, pos, nbTimesteps, s, train = train, debug = debug)
    
    return res


class OneStepSimulator():
    def __init__(self, model):
        self.model = model

    def simulate(self, sim, device = DEVICE):

        hist = np.zeros_like(sim)
        self.model.eval
        with torch.no_grad():
            x, y, attr, inds = processSimulation(sim)

            for t in range(len(x)):

                pos = x[t][:, :2].to(device)

                data = Data(x = x[t][:, 2:], y = y[t] , edge_attr = attr[t], edge_index = inds[t])

                data = normalizeGraph(data).to(DEVICE)


                pred = self.model(data)
                nextPose = pos + pred

                pred[:, 0] = torch.where((nextPose[:, 0] < -BOUNDARY) | (nextPose[:, 0] > BOUNDARY), -pred[:, 0], pred[:, 0])
                pred[:, 1] = torch.where((nextPose[:, 1] < -BOUNDARY) | (nextPose[:, 1] > BOUNDARY), -pred[:, 1], pred[:, 1])

                nextPose = pos + pred

                hist[t] = nextPose.cpu().numpy()


        return hist