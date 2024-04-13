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


MIN_X = cfg['normalization']['distance']['minDistance']
MAX_X = cfg['normalization']['distance']['maxDistance']
MIN_Y = cfg['normalization']['distance']['minDistance']
MAX_Y = cfg['normalization']['distance']['maxDistance']



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


###############
# Simulator
###############

class NNSimulator():
    def __init__(self, model):

        self.model = model


    def runSim(self, T, state):

        return genSim(self.model, T, state, train = False)
        
        
def genSim(model, T, state, pos, train = True, normInit = True):
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

    # if first state is not normalized do it
    #normInit

    # if training, keep the computation graph
    if train:
        
        yList = []  # list of speeds

        for i in range(T):

            # one-step transition
            y = model(state)
            yList.append(torch.unsqueeze(y, dim = 0))

            # update of the states and positions
            #with torch.no_grad():  # ?

            if OUTPUT_TYPE == 'speed':
                # Effect of boundary conditioned by previous position
                y = boundaryEffect(hist[-1][0], y, BOUNDARY )               
                state, position = updateData(state, hist[-1][0], y)


            elif OUTPUT_TYPE == 'acceleration':
                pass

            hist.append(torch.unsqueeze(position, dim = 0))

        return torch.cat(hist, dim = 0), torch.cat(y, dim = 0)       # [T, N, 2], [T, N, 2]
    
    # if not training, do not keep the comptutation graph
    else:
        model.eval()
        with torch.no_grad():

            for i in tqdm(range(T)):

                # one-step transition
                y = model(state)

                if OUTPUT_TYPE == 'speed':
                    # Effect of boundary conditioned by previous position
                    y = boundaryEffect(hist[-1][0], y, BOUNDARY )
                    state, position = updateData(state, hist[-1][0], y)

                elif OUTPUT_TYPE == 'acceleration':
                    pass

                hist.append(torch.unsqueeze(position, dim = 0))

            return torch.cat(hist, dim = 0)
        
        model.train()


def boundaryEffect(x, v, boundary = BOUNDARY):
    
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
    
    # normalize it for the state
    xState = normalizeCol(nextPos[:,0], MIN_X, MAX_X).view(-1, 1)
    yState = normalizeCol(nextPos[:,1], MIN_Y, MAX_Y).view(-1, 1)
    nextState = torch.cat((xState, yState), dim = -1)
    
    
    
    # update state
    R = prevState[:, -1]    # radius [N]

    # get new state by concatenation
    newS = torch.cat((nextState, speed), dim = -1)
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

def optimized_getGraph(mat_t, threshold=THRESHOLD_DIST):
    """
    Optimized function to compute the graph for PyTorch Geometric.

    Args:
    -----
        - `mat_t` (np.array): 2D np array (matrix at a given timestep)
        - `threshold` (float): Distance threshold for connecting vertices

    Returns:
    --------
        - `distList` (torch.Tensor): Tensor of distances and direction cosines.
        - `indices` (torch.Tensor): Tensor of graph indices.
    """
    num_points = mat_t.shape[0]
    # Expand dims to broadcast and compute all pairwise distances
    mat_expanded = np.expand_dims(mat_t, 1)  # Shape: [N, 1, 2]
    all_dists = np.sqrt(np.sum((mat_expanded - mat_t)**2, axis=2))  # Shape: [N, N]

    # Identify pairs below the threshold, excluding diagonal
    ix, iy = np.triu_indices(num_points, k=1)
    valid_pairs = all_dists[ix, iy] < threshold

    # Filter pairs by distance threshold
    filtered_ix, filtered_iy = ix[valid_pairs], iy[valid_pairs]
    distances = all_dists[filtered_ix, filtered_iy]

    # Calculate direction vectors and angles
    direction_vectors = mat_t[filtered_iy] - mat_t[filtered_ix]
    angles = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0])

    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    # Normalize distances and create distance vectors
    normalized_dists = normalizeCol(distances, MIN_DIST, MAX_DIST)
    dist_vectors = np.stack([normalized_dists, cos_theta, sin_theta], axis=1)

    # Double entries for bidirectional edges
    doubled_indices = np.vstack([np.stack([filtered_ix, filtered_iy], axis=1),
                                 np.stack([filtered_iy, filtered_ix], axis=1)])
    doubled_dist_vectors = np.vstack([dist_vectors, dist_vectors * [1, -1, -1]])

    # Convert to tensors
    indices_tensor = torch.tensor(doubled_indices.T, dtype=torch.long)
    dist_tensor = torch.tensor(doubled_dist_vectors, dtype=torch.float)

    return dist_tensor, indices_tensor



def updateData(prevState, prevPose, speed, device = DEVICE, threshold = THRESHOLD_DIST):

    newS, nextPose = updateState(prevState.x,prevPose, speed)

    newGraph, newInds = optimized_getGraph(nextPose.cpu().detach().numpy().copy())

    return Data(x = newS, edge_index = newInds, edge_attr = newGraph).to(device), nextPose

