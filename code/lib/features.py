import numpy as np
import os
import yaml
import torch
from norm import normalizeCol

# delete last elements because can't predict for it
# can't keep the NB_ROLLOUT elements because can't do it entirely for them
# delete first element because can't know the v_{0-1}

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 
    
nomalization = cfg['normalization']
features = cfg['feature']


NB_HIST = features['nbHist']  # number of repetitions fo lagged values
THRESHOLD_DIST = features['distGraph']
NUMBER_ROLLOUT = features['nbRolloutOut']
FEATURE_SHAPE = features['inShape']
EDGE_SHAPE = features['edgeShape']


# noramlize distances
MIN_DIST = nomalization['distance']['minDistance']
MAX_DIST = nomalization['distance']['maxDistance']

MIN_X = nomalization['position']['minPos']
MAX_X = nomalization['position']['maxPos']
MIN_Y = nomalization['position']['minPos']
MAX_Y = nomalization['position']['maxPos']


###############
# Outputs
###############


def predictSpeeds(mat:np.array)->list:
    """
    Function to create the output predicted by the NN

    Args:
    -----
        - `mat`

    Returns:
    --------
        list of speeds
    """
    time = mat.shape[0]
    speeds = getSpeeds(mat)
    y=[]
    
    # element i ==> speed i->i+1
    for i in range(1, time - NUMBER_ROLLOUT):
        v = None
        for t in range(NUMBER_ROLLOUT):
            nb = i + t
            
            if v is None:
                v = torch.from_numpy(speeds[nb, :, :])
            else:
                v = torch.cat((v,torch.from_numpy(speeds[nb, :, :])), dim = -1)
        
        y.append(v)
        
    return y


###############
# Node features
###############


def getSpeeds(mat:np.array)->np.array:
    """ 
    Function in order to compute the speeds for each cell
    
    time i ==> speed = pos[i+1] - pos[i] (forward speeds)
    
    Args:
    -----
        - `mat`[Time, #Nodes, 2]: positions of the different cells through time
    
    Ouptut:
    -------
        A np.array [Time-1, Node, 2] with the two speeds (v_x, v_y) at each row
    
    """
    T = mat.shape[0]
    
    inds0 = np.arange(0, T-1)
    inds = np.arange(1, T)
    
    # [i, n, :] = speed for node n between i->i+1
    # ok for predictions
    return mat[inds, :, :] - mat[inds0, :, :]



def getSpeedFeatures(mat: np.array, nb: int)->np.array:
    """ 
    Function to get the concatenation of speeds as features

    Args:
    -----
        - `mat` (np.array): positions of the different cells through time [Time, #Nodes, 2]
        - `nb` (int):  repetition of the lagged values

    Returns:
        res (np.array): array containing the features [v_x, v_y] x nb
    """

    # adding speed norm
    res = None
    speeds = getSpeeds(mat)

    # uncomment following to activate speed norms
    #speedNorms = np.sqrt(speeds[:, :, 0] ** 2 + speeds[:, :, 1] ** 2)
    #speeds = np.concatenate((speeds, speedNorms[:, :, np.newaxis]), axis = -1)

    for i in range(nb):
        s = np.zeros((mat.shape[0], mat.shape[1], speeds.shape[-1]))
        
        if i != 0:
            s[(i+1):, :, :] = speeds[:-i, :, :]
            res = np.concatenate((res, s), axis = -1)
        else:
            s[(i+1):, :, :] = speeds
            res = s
            
    return res


def addParams(mat:np.array, params:np.array)-> np.array:
    """ 
    Function to add some given parameters to all
    instances of the arary (3 dims) (add it in last dimension)

    NOTE: used for adding the radius of the cells (need change if different radius)
    Args:
    ----
        - `mat` (np.array): feature array
        - `params` (np.array): array to add at the end of 3rd dimension

    Retunrs:
    --------
        concatenated array
    """

    p = np.repeat(params.reshape(1, -1), mat.shape[0] * mat.shape[1], axis = 0).reshape(mat.shape[0], mat.shape[1], -1)
    res = np.concatenate((mat, p), axis = -1)
    return res


def getFeatures(mat: np.array, params:np.array, nb:int = NB_HIST)->tuple:
    """ 
    Function that allows to compute the features of the nodes 
    and the structure of the graph
    
    Args:
    -----
        - `mat`[Time x #Nodes x 2]: positions of the different cells through time
        - `params`: additional features (already normalized)
        - `nb`: number of past timesteps concatenated in the macrostate of the cells
    
    Output:
    -------
        - a list with the features of the nodes.
            Each element correspond to a timestep and is a torch.tensor()
            of shape [#Nodes x features space]
        - a list of the next positions (to predict)
    """
        
    # output type
    y = predictSpeeds(mat)


    yB = []
    for i in range(len(y)):
        yB.append(y[i])
    

    # features
    x = getSpeedFeatures(mat, nb)
    
    # add the parameters
    x = addParams(x, params)    

    vectnodes = []

    for i in range(1, x.shape[0]-NUMBER_ROLLOUT):
        vectnodes.append(torch.from_numpy(x[i]).to(torch.float))
    
    return vectnodes, yB


###############
# Edges features
###############


def distance(v1, v2):
    return np.linalg.norm(v1-v2)


def getEdges2(mat:np.array,threshold:float = THRESHOLD_DIST)->tuple:
    """
    Function to get the edges and the labels for a graph
    It will select the closest neighbors to a given nodes as 
    linked nodes
    
    Args:
    -----
    - `mat`[Time x #Nodes x 2]: positions of the different cells through time
    - `threshold`: threshold for the computation of distances

    Returns:
        tuple of:
        indices of edges
        values of edges
    """
    
    resD = []
    resInd = []
    
    # start at index T-1 until one before the end
    # last element not in inputs
    for t in range(1, mat.shape[0] - NUMBER_ROLLOUT):
        distList = []
        indices = []

        for i in range(mat.shape[1]):
            for j in range(i+1, mat.shape[1]):

                # compute the distance between cell at given timestep
                dist = distance(mat[t, i, :], mat[t, j, :])
                
                
                if dist < threshold:
                    #adj[i,j] = 1
                    #adj [j, i] = 1
                    
                    indices += [[i, j], [j, i]]

                    direction_vector = mat[t, j, :] - mat[t, i, :]
                    
                    dist = normalizeCol(dist.copy(), MIN_DIST, MAX_DIST)
                                                            
                    distList.append(torch.tensor([dist, direction_vector[0], direction_vector[1]], dtype=torch.float).unsqueeze(0))
                    distList.append(torch.tensor([dist, direction_vector[1], direction_vector[0]], dtype=torch.float).unsqueeze(0))
        
        indices = torch.tensor(indices)
        indices = indices.t().to(torch.long).view(2, -1)
        distList = torch.cat(distList, dim = 0)
        
        resD.append(distList)
        resInd.append(indices)
    return resInd, resD


def getEdges(mat:np.array,threshold:float = THRESHOLD_DIST)->tuple:
    """
    Function to get the edges and the labels for a graph
    It will select the closest neighbors to a given nodes as 
    linked nodes
    
    Args:
    -----
    - `mat`[Time x #Nodes x 2]: positions of the different cells through time
    - `threshold`: threshold for the computation of distances

    Returns:
        tuple of:
        indices of edges
        values of edges
    """
    
    resD = []
    resInd = []

    # start at index T-1 until one before the end
    # last element not in inputs
    
    for t in range(1, mat.shape[0] - NUMBER_ROLLOUT):

        vd, iInd = optimized_getGraph(mat[t, :, :], threshold=THRESHOLD_DIST)
        
        resD.append(vd)
        resInd.append(iInd)
    return resInd, resD


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

    # Double entries for bidirectional edges
    doubled_indices = np.vstack([np.stack([filtered_ix, filtered_iy], axis=1),
                                 np.stack([filtered_iy, filtered_ix], axis=1)])
    
    doubled_dist_vectors = np.vstack([np.stack([normalized_dists, direction_vectors[:, 0], direction_vectors[:, 1]], axis=1),
                                      np.stack([normalized_dists, direction_vectors[:, 1], direction_vectors[:, 0]], axis=1)])

    # Convert to tensors
    indices_tensor = torch.tensor(doubled_indices.T, dtype=torch.long)
    dist_tensor = torch.tensor(doubled_dist_vectors, dtype=torch.float)

    return dist_tensor, indices_tensor


# new getGraph for NNSimulator