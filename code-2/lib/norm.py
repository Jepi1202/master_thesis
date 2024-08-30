import numpy as np
from torch_geometric.utils import sort_edge_index

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



def normalizeGraph(graph):
    # do not change the features of the nodes
    # since use cos and sin, no real need for normalization
    #: might still be useful for the distance in the edges though
    
    graph.edge_index, graph.edge_attr = sort_edge_index(graph.edge_index, graph.edge_attr)

    return graph