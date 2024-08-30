import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import sort_edge_index

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')


from measure import plotStdMessage





def findIndices(message, nb = 5):
    
    stdv=plotStdMessage(message)
    plt.close()

    inds = np.argsort(stdv)
    # change of the order
    return np.flip(inds[-nb:])


########################################


def getForces(graph, k, epsilon):
    inds = graph.edge_index
    #print(inds)

    messages = []

    for i in range(inds.shape[1]):
        messages.append(calculate_interaction(graph.edge_attr[i, 0],
                                              torch.tensor([graph.edge_attr[i, 0] * graph.edge_attr[i, 1], graph.edge_attr[i, 0] * graph.edge_attr[i, 2]]),
                                              k = k,
                                              epsilon=epsilon
                                              ).cpu().detach().numpy())
        
    return messages



def getForces2(graphs, k, epsilon):

    res = None

    for graph in graphs:
        interaction = calculate_interaction(graph.edge_attr[i, 0],
                                            torch.tensor([graph.edge_attr[i, 0] * graph.edge_attr[i, 1], graph.edge_attr[i, 0] * graph.edge_attr[i, 2]]),
                                            k = k,
                                            epsilon=epsilon
                                            ).cpu().detach().numpy()



def calculate_interaction(dist, rij, k, epsilon, radii = 1.0):
    """
    Given the vectors ri and rj, compute the force between them
    """


    r = dist

    # Combined radius of both particles (assume unit radii for now)
    #bij = 2.0                       # Ri + Rj 
    bij = radii + radii

    if r < bij*(1 + epsilon):
        force = k*(r - bij)*rij/r  
    elif r < bij*(1 + 2*epsilon):
        force = -k*(r - bij - 2*epsilon*bij)*rij/r
    else:
        force = torch.tensor([0.0, 0.0])
    return force


########################################


def getGroundTruth(data, k, epsilon):

    res = []

    for graph in data:

        res.extend(getForces(graph, k, epsilon))


    return np.array(res)

########################################


def getPrediction(model, data, inds = None):
    res = None

    for graph in data:

        if inds is None:
            messages = model.message(graph).cpu().detach().numpy()
        else:   
            messages = model.message(graph).cpu().detach().numpy()[:, inds]
        
        
        if res is None:
            res = messages
        else:
            res = np.vstack((res, messages))

    return res


def get_messages_model(model, data, nbMax:int = 2):
    message = getPrediction(model, data, inds = None)


    if nbMax is None:
        return message
    
    inds = findIndices(message, nb = nbMax)

    return message[:, inds]

########################################


def get_sum_messages_model(model, data, nbMax:int = 2):
    res = None

    for graph in data:

        messages = model.sum_message(graph).cpu().detach().numpy()
        
        
        if res is None:
            res = messages
        else:
            res = np.vstack((res, messages))

    if nbMax is None:
        return res

    inds = findIndices(res, nb = nbMax)
    res = res[:, inds]



    return res

########################################


def getInputs(data):
    res = None

    for graph in data:

        inputs = graph.x.cpu().detach().numpy()

        if res is None:
            res = inputs
        else:
            res = np.vstack((res, inputs))

    return res

########################################



def getEdges(data):
    res = None

    for graph in data:

        edges = graph.edge_attr.cpu().detach().numpy()

        if res is None:
            res = edges
        else:
            res = np.vstack((res, edges))

    return res


########################################


def getOutput(mdoel, data):
    res = None

    for graph in data:

        y = mdoel(graph).cpu().detach().numpy()

        if res is None:
            res = y
        else:
            res = np.vstack((res, y))

    return res


########################################


def getgtOutput(data):
    res = None

    for graph in data:

        y = graph.y.cpu().detach().numpy()

        if res is None:
            res = y
        else:
            res = np.vstack((res, y))

    return res



########################################

"""
def get_weights(model, data, nbDim = 2, nbMax = 2, pySr_data = False):
    
    res = None
    resEdges = None

    if pySr_data:

        for graph in data:

            weights, sha = model.get_weights(graph)
            weights = weights.cpu().detach().numpy()
            weights.reshape(sha)

            if res is None:
                res = weights
                shape_edges =  graph.edge_attr.shape
                resEdges = graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()
            else:
                res = np.vstack((res, weights))
                shape_edges =  graph.edge_attr.shape
                v = graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()
                resEdges = np.vstack((resEdges, graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()))

        return res, resEdges

    else:
        for graph in data:

            weights, sha = model.get_weights(graph)
            weights = weights.cpu().detach().numpy()

            if res is None:
                res = weights

            else:
                res = np.vstack((res, weights))

            if resEdges is None:
                resEdges = graph.edge_attr

            else:
                resEdges = np.vstack((resEdges, graph.edge_attr))

        return res, resEdges

"""
def get_weights(model, data, nbDim = 2, nbMax = 2, pySr_data = False):
    
    res = None
    resEdges = None

    if pySr_data:

        for graph in data:

            weights, sha = model.get_weights(graph)
            weights = weights.cpu().detach().numpy()
            weights.reshape(sha)

            if res is None:
                res = weights
                shape_edges =  graph.edge_attr.shape
                resEdges = graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()
            else:
                res = np.vstack((res, weights))
                shape_edges =  graph.edge_attr.shape
                v = graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()
                resEdges = np.vstack((resEdges, graph.edge_attr.repeat(1, nbDim).reshape(shape_edges[0] * nbDim, shape_edges[1]).cpu().detach().numpy()))

        return res, resEdges

    else:
        for graph in data:

            weights, sha = model.get_weights(graph)
            weights = weights.cpu().detach().numpy()

            if res is None:
                res = weights

            else:
                res = np.vstack((res, weights))

            if resEdges is None:
                resEdges = graph.edge_attr

            else:
                resEdges = np.vstack((resEdges, graph.edge_attr))

        return res, resEdges




def get_sum_gam_weights(model, data):

    resWeights = None

    for graph in data:
        atr = graph.edge_attr
        index_full = graph.edge_index

        index_full, atr = sort_edge_index(index_full, atr)

        w_sum = model.GNN.message(None, None, atr, index_full).cpu().detach().numpy()


        if resWeights is None:
            resWeights = w_sum

        else:
            resWeights = np.vstack((resWeights, w_sum))


    return resWeights