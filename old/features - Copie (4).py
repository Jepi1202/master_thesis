import numpy as np
import torch
from torch_geometric.data import Data

# [x, y, v_x, v_y, v] R
# [d, cos, sin, Ri, Rj]

# Position norm
MIN_X = -150
MAX_X = 150
MIN_Y = -150
MAX_Y = 150

# speeds norm
# not used ...
#MIN_V_X = -150
#MAX_V_X = 150
#MIN_V_Y = -150
#MAX_V_Y = 150


# noramlize distances
MIN_DIST = 0
MAX_DIST = 150


# number of repetitions fo lagged values
nbHist = 2
THRESHOLD_DIST = 40.


# FEATURES SHAPE
#TODO
FEATURE_SHAPE = 2*4+1

###############
# Outputs
###############


def predictDisplacement(mat:np.array)->list:
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
    y=[]
    
    # element i ==> speed i->i+1
    for i in range(1, time):
        y.append(torch.from_numpy(mat[i, :, :] - mat[i-1, :, :]).to(torch.float))
        
    return y




###############
# Node features
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


def getAngles(speeds:np.array)->np.array:
    """ 
    Allows to compute the sin and cosine from the speeds
    
    Args:
    -----
        - `speeds` (np.array): speeds of the cells [T-1, 2]

    Returns:
    --------
        the speeds cosine and sine [T-1, 2]
    """
    
    mat = np.zeros_like(speeds)
    theta = np.arctan2(speeds[:, 1], speeds[:, 0])
    sinu = np.sin(theta)
    cosi = np.cos(theta)
    
    if np.any(np.isnan(sinu)):
        print(f'nan sine')
        
    if np.any(np.isnan(cosi)):
        print(f"nan cos")
        
    mat[:, 0] = cosi
    mat[:, 1] = sinu
    
    return mat



def getSpeeds(mat:np.array)->np.array:
    """ 
    Function in order to compute the speeds for each cell
    
    id i ==> speed = pos[i+1] - pos[i] (forward speeds)
    
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
    return mat[inds, :, :] - mat[inds0, :, :]




def getRepFeatures(mat: np.array, nb: int)->np.array:
    """ 
    Function to associate the positions and the speeds in the features

    Args:
    -----
        - `mat` (np.array): positions of the different cells through time [Time, #Nodes, 2]
        - `nb` (int):  repetition of the lagged values

    Returns:
        res (np.array): array containing the features [x, y, v_x, v_y, v] x nb
    """
    # adding speed norm
    res = None
    speeds = getSpeeds(mat)

    # uncomment following to activate speed norms
    #speedNorms = np.sqrt(speeds[:, :, 0] ** 2 + speeds[:, :, 1] ** 2)
    #speeds = np.concatenate((speeds, speedNorms[:, :, np.newaxis]), axis = -1)
    

    # apply normalization on position
    mat[:, :, 0] = normalizeCol(mat[:, :, 0], MIN_X, MAX_X)
    mat[:, :, 1] = normalizeCol(mat[:, :, 1], MIN_Y, MAX_Y)

    for i in range(nb):
        s = np.zeros((mat.shape[0], mat.shape[1], speeds.shape[-1]))
        p = np.zeros_like(mat)
        
        if i != 0:
            s[(i+1):, :, :] = speeds[:-i, :, :]
            p[i:, :, :] = mat[:-i, :, :]
            r = np.concatenate((p, s), axis = -1)
            res = np.concatenate((res, r), axis = -1)
        else:
            s[(i+1):, :, :] = speeds
            p[i:, :, :] = mat
            res = np.concatenate((p, s), axis = -1)
            
     
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



def getFeatures(mat: np.array, params:np.array, nb:int = nbHist)->tuple:
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
        
    # output
    y = predictDisplacement(mat)

    yB = []
    for i in range(len(y)):

            yB.append(y[i])
    
    # features
    x = getRepFeatures(mat, nb)
        
        
    #s = np.zeros_like(mat)
    #speeds = getSpeeds(mat)
    #s[1:, :] = speeds
    
    #angleSpeeds = getAngles(s)
    
    #x = np.concatenate((x, angleSpeeds), axis= -1)
    
    
    x = addParams(x, params)    

    vectnodes = []
    # last one, don't have the output
    for i in range(x.shape[0]-1):
        vectnodes.append(torch.from_numpy(x[i]).to(torch.float))
    
    return vectnodes, yB


###############
# Edges features
###############


def distance(v1, v2):
    return np.linalg.norm(v1-v2)


def getGraph(mat_t, mat2, threshold = THRESHOLD_DIST):
    """ 
    Function to compute the graph for pytorch geometric

    Args:
    -----
        - `mat_t` (np.array): 2D np array (mat at a given timestep)
        - `mat2` (np.array): normalized version of matrix mat_t
        - `threshold` (int): 

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
                #adj[i,j] = 1
                #adj [j, i] = 1
                
                indices += [[i, j], [j, i]]

                direction_vector = mat_t[j, :] - mat_t[i, :]
                angle = np.arctan2(direction_vector[1], direction_vector[0])
                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)

                
                dist = normalizeCol(dist, MIN_DIST, MAX_DIST)
                                                        
                distList.append(torch.tensor([dist, cos_theta, sin_theta, mat2[i, 0], mat2[i, 1], mat2[ j, 0], mat2[j, 1]], dtype=torch.float).unsqueeze(0))
                distList.append(torch.tensor([dist, -cos_theta, -sin_theta, mat2[j, 0], mat2[j, 1], mat2[i, 0], mat2[i, 1] ], dtype=torch.float).unsqueeze(0))

    indices = torch.tensor(indices)
    indices = indices.t().to(torch.long).view(2, -1)
    distList = torch.cat(distList, dim = 0)

    return distList, indices




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

    mat2 = mat.copy()
    mat2[:, :, 0] = normalizeCol(mat2[:, :, 0], MIN_X, MAX_X)
    mat2[:, :, 1] = normalizeCol(mat2[:, :, 1], MIN_Y, MAX_Y)
    
    # start at index T-1 until one before the end
    # last element not in inputs
    for t in range(mat.shape[0]-1):
        
        distList, indices = getGraph(mat[t, :, :], mat2[t, :, :], threshold)
        
        resD.append(distList)
        resInd.append(indices)
    return resInd, resD




##########################################
# Simulator functions
##########################################
"""
class Hist():
    def __init__(self, nbHist, x):
        # Initial conditions

        # positions
        self.prevPos = np.zeros((nbHist, 2))
        self.prevPos[-1, 0] = normalizeCol(x[0], MIN_X, MAX_X)
        self.prevPos[-1, 1] = normalizeCol(x[1], MIN_Y, MAX_Y)

        # speeds
        self.prevSpeeds = np.zeros((nbHist, 2))
        
        self.feats = np.concatenate((self.prevPos, self.prevSpeeds), axis=-1)


    def update(self, xNew, yNew, v_x_new, v_y_new):

        newPos = np.zeros(self.prevPos.shape[0]+1, 2)
        newSpeeds = np.zeros(self.prevSpeeds.shape[0]+1, 2)

        newPos[:-1, :] = self.prevPos
        newSpeeds[:-1, :] = self.prevSpeeds

        xNew = normalizeCol(xNew, MIN_X, MAX_X)
        yNew = normalizeCol(yNew, MIN_Y, MAX_Y)

        newPos[-1, 0] = xNew
        newPos[-1, 1] = yNew
        newSpeeds[-1, 0] = v_x_new
        newSpeeds[-1, 1] = v_y_new

        self.prevPos = newPos
        self.prevSpeeds = newSpeeds

        self.feats = np.concatenate((self.prevPos, self.prevSpeeds), axis=-1)


    def getFeatures(self, params, shape = FEATURE_SHAPE, nbHist = 2):

        vals = self.feats[-nbHist:, :].reshape(-1)
        vals = np.concatenate((vals, params), axis=-1)

        return vals

"""

class Hist():
    def __init__(self, nbHist, x):

        self.nbHist = nbHist
        self.h = x

    def update(self, y):
        self.h = np.concatenate((self.h,y), axis = 0)

    def getFeatures(self, params, threshold = THRESHOLD_DIST):
        if self.h.shape[0] < self.nbHist:
            res = np.zeros((self.nbHist, self.h.shape[1],2))
            L = self.h.shape[0]
            res[-L:,:] = self.h

            return getFeatures(res, params, threshold)


        return getFeatures(self.h, params, threshold)


class simulNet():
    def __init__(self, network):
        self.net = network
        self.hist = None


    def simulate(self, x, t, nbHist, params, threshold = THRESHOLD_DIST):
        if self.hist is None:
            self.hist = Hist(nbHist, x)

        # x: [tau, N, 2]
        x2 = np.zeros_like(x)
        xOld = x

        for i in range(t):

            # get graph
            x2[:, :, 0] = normalizeCol(x[:, :, 0], MIN_X, MAX_X)
            x2[:, :, 1] = normalizeCol(x[:, :, 1], MIN_Y, MAX_Y)
            attr, indices = getGraph(x[-1, :, :], x2[-1, :, :], threshold)

            # get features
            f = self.hist.getFeatures(params)

            # apply network

            data = Data(x=f, 
                        edge_index=indices,
                        edge_attr=attr,
                        y=None
                        ) 
            
            s = self.net(data)

            # get next positions from predicted speed

            xNew = xOld + s