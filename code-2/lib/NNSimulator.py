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


print('fndjngkjdfs --NN-sim')
print('update')

###############
# Simulator
###############

class NNSimulator():
    def __init__(self, model):
        self.model = model


    def runSim(self, T, state, pos, dt_scale, debug = None, train = False, device = DEVICE):
        return genSim(self.model, T, state, pos, train = train, debug = debug, dt_scale = dt_scale, device = device)
        
        
def genSim(model, T, state, pos, radius=None, train = True, debug = None, dt_scale = 1, device = DEVICE):
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
    #radius = np.ones(pos.shape[0]) * R_PARAM      #############################################
    radius = torch.ones(pos.shape[0], device = device) * R_PARAM

    pos = pos.to(device)
    hist = [torch.unsqueeze(pos, dim = 0).to(device)]          # [ [1, N, 2] ]
    
    #print(state)

    # if training, keep the computation graph
    if train:

        yList = []  # list of speeds

        for i in range(T):
            
            # one-step transition

            if debug is not None:
                y = model(debug[i]).to(device)
            else:
                y = model(state)

            # update of the states and positions
            #with torch.no_grad():  # ?

            if OUTPUT_TYPE == 'speed':
                # Effect of boundary conditioned by previous position
                nPose = hist[-1][0] + y * dt_scale
                vNext = boundaryEffect(nPose, y, BOUNDARY )
                
            #elif OUTPUT_TYPE == 'acceleration':
            #    v = state.x[:, :2]
            #    vNext = v + y
            #    nPose = hist[-1][0] + vNext
            #    vNext = boundaryEffect(nPose, vNext, BOUNDARY)

            yList.append(vNext)
            state, position = updateData(state, hist[-1][0], vNext, radius = radius, dt_scale = dt_scale, device = device)
            hist.append(torch.unsqueeze(position, dim = 0))

        return torch.cat(hist, dim = 0), torch.stack(yList)       # [T, N, 2], [T, N, 2]

    # if not training, do not keep the comptutation graph
    else:
        model.eval()
        with torch.no_grad():

            for i in tqdm(range(T)):

                # one-step transition
                if debug is not None:
                    y = model(debug[i]).to(device)
                else:
                    y = model(state)
                    #print(y)


                if OUTPUT_TYPE == 'speed':
                    # Effect of boundary conditioned by previous position
                    nPose = hist[-1][0] + y * dt_scale
                    vNext = boundaryEffect(nPose, y, BOUNDARY )

                #elif OUTPUT_TYPE == 'acceleration':
                #    v = state.x[:, :2]
                #    vNext = v + y
                #    nPose = hist[-1][0] + vNext
                #    vNext = boundaryEffect(nPose, vNext, BOUNDARY )               

                state, position = updateData(state, hist[-1][0], vNext, radius = radius, dt_scale = dt_scale, device = device)
                hist.append(torch.unsqueeze(position, dim = 0))

        return torch.cat(hist, dim = 0)


def boundaryEffect(x, v, boundary:float = BOUNDARY):
    v[:, 0] = torch.where((x[:, 0] < -BOUNDARY) | (x[:, 0] > BOUNDARY), -v[:, 0], v[:, 0])
    v[:, 1] = torch.where((x[:, 1] < -BOUNDARY) | (x[:, 1] > BOUNDARY), -v[:, 1], v[:, 1])
    
    return v


def updateState(prevState, prevPos, speed, dt_scale):
    """ 
    Updates the prevStates into the following one

    Args:
    -----
        - `prevState`:previous state[[x, y, v_x, v_y]_t xN R]   [N, D]
        - `prevPos`: previous position
        - `speed`: speed [v_x, v_y]    [N, 2]
    """

    # get the next positions
    nextPos = prevPos + speed * dt_scale
        
    #print(speed)

    # get new state by concatenation
    nextState = torch.cat((speed, prevState[:, :-2]), dim = -1)
    #nextState = torch.cat((prevState[:, 2:], speed), dim = -1)
    
    #print(nextState)

    return nextState, nextPos


def updateData(prevState, prevPose, speed, radius, device = DEVICE, threshold = THRESHOLD_DIST , dt_scale = 1):

    newS, nextPose = updateState(prevState.x,prevPose, speed, dt_scale = dt_scale)

    #newGraph, newInds = optimized_getGraph(nextPose.cpu().detach().numpy().copy(), radius = radius) ############
    
    newGraph, newInds = optimized_getGraph_th(nextPose, radius = radius, device = device)

    data = Data(x = newS, edge_index = newInds, edge_attr = newGraph)
    
    data = normalizeGraph(data)

    return data.to(device), nextPose


def getSimulationVideo(model:torch.tensor, initPos:torch.tensor, nbTimesteps:int, initState:torch.tensor, train = False, debug = None, dt_scale = 1, device = DEVICE) -> torch.tensor:
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

    res = simulator.runSim(nbTimesteps, initState, initPos, train = train, debug = debug, dt_scale = dt_scale, device = device)

    #create_simulation_video_cv2(res, outputPath, fps = 10, size = (600,600))

    return res



def getSimulationData(model:torch.tensor, nbTimesteps:int, d:np.array, i = 5, display =True, train = False, debug = None, radius = None, dt_scale = 1, device = DEVICE) -> torch.tensor:
    if radius is None:
        radius = np.ones(d.shape[1]) * R_PARAM

    x, y = getFeatures(d.copy(), nb = 4)
    attr, inds = optimized_getGraph(d[i, :, :].copy(), radius = radius)
    s = Data(x = x[i-1][: , 2:], edge_attr = attr, edge_index = inds)

    s = normalizeGraph(s).to(device)

    res = getSimulationVideo(model, torch.from_numpy(d[i, :, :].copy()).float(), nbTimesteps, s, train = train, debug = debug, dt_scale = dt_scale, device = device)
    
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
    
    
    
    
def optimized_getGraph_th(mat_t, radius, threshold=6, device='cuda'):
    """
    Optimized function to compute the graph for PyTorch Geometric using PyTorch tensors.

    Args:
    -----
        - `mat_t` (torch.Tensor): 2D torch tensor (matrix at a given timestep).
        - `radius` (torch.Tensor): 1D torch tensor of radii.
        - `threshold` (float): Distance threshold for connecting vertices.
        - `device` (str): The device to perform computations on ('cpu' or 'cuda').

    Returns:
    --------
        - `distList` (torch.Tensor): Tensor of distances and direction cosines.
        - `indices` (torch.Tensor): Tensor of graph indices.
    """
    with torch.no_grad():
        # Move tensors to the specified device
        mat_t = mat_t.to(device)
        radius = radius.to(device)
        
        num_points = mat_t.shape[0]
        
        # Expand dims to broadcast and compute all pairwise distances
        mat_expanded = mat_t.unsqueeze(1)  # Shape: [N, 1, 2]
        all_dists = torch.sqrt(torch.sum((mat_expanded - mat_t) ** 2, dim=2))  # Shape: [N, N]

        # Identify pairs below the threshold, excluding diagonal
        ix, iy = torch.triu_indices(num_points, num_points, offset=1, device=device)
        valid_pairs = all_dists[ix, iy] < threshold

        # Filter pairs by distance threshold
        filtered_ix = ix[valid_pairs]
        filtered_iy = iy[valid_pairs]
        distances = all_dists[filtered_ix, filtered_iy]

        # Calculate direction vectors
        direction_vectors = (mat_t[filtered_iy] - mat_t[filtered_ix]) / distances.unsqueeze(1)

        radii_i = radius[filtered_ix]
        radii_j = radius[filtered_iy]

        # Double entries for bidirectional edges
        doubled_indices = torch.cat([
            torch.stack([filtered_ix, filtered_iy], dim=1),
            torch.stack([filtered_iy, filtered_ix], dim=1)
        ], dim=0)
        
        doubled_dist_vectors = torch.cat([
            torch.stack([distances, direction_vectors[:, 0], direction_vectors[:, 1], radii_i, radii_j], dim=1),
            torch.stack([distances, -direction_vectors[:, 0], -direction_vectors[:, 1], radii_j, radii_i], dim=1)
        ], dim=0)

        # Convert to tensors (already in tensor form, but ensuring shape)
        indices_tensor = doubled_indices.T.to(torch.long)
        dist_tensor = doubled_dist_vectors.to(torch.float)

        return dist_tensor, indices_tensor
