import numpy as np
import sys

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

import simulation as sim 
from torch_geometric.data import Data
from norm import normalizeGraph
import features as ft


class Parameters_Simulation():
    def __init__(self):
        self.dt = 0.001
        self.v0 = 60
        self.k = 70
        self.epsilon = 0.5
        self.tau = 3.5
        self.R = 1
        self.N = 200
        self.boundary = 100

        self.nbStep = 150


class Initial_Conditions():
    def __init__(self, pos: np.array, angles: np.array):
        self.pos = pos
        self.angles = angles


def create_initial_cond(params: Parameters_Simulation):
    
    N = params.N
    v0 = params.v0
    k = params.k
    eps = params.epsilon
    tau = params.tau
    R = params.R
    dt = params.dt
    nbStep = params.nbStep
    boundary = params.boundary

    lim = 0.85 * boundary

    xPos = np.linspace(-lim, lim, 10)
    yPos = np.linspace(-lim, lim, 10)
    gridX, gridY = np.meshgrid(xPos, yPos)
    delta = np.random.uniform(0, 7, gridX.shape + (2,))

    gridX2 = gridX + delta[:, :, 0]
    gridY2 = gridY + delta[:, :, 1]


    pos = np.stack([gridX.ravel(), gridY.ravel()], axis=1)
    pos_perturbed = np.stack([gridX2.ravel(), gridY2.ravel()], axis=1)

    pos = np.concatenate([pos, pos_perturbed], axis=0)

    angles = np.random.rand(pos.shape[0]) * 2 * np.pi


    out = Initial_Conditions(pos, angles)

    return out




def get_data(params: Parameters_Simulation, init_cond = None):
    N = params.N
    v0 = params.v0
    k = params.k
    eps = params.epsilon
    tau = params.tau
    R = params.R
    dt = params.dt
    nbStep = params.nbStep
    boundary = params.boundary

    ## get initial conditions
    if init_cond is None:
        init_cond = create_initial_cond(params)
    

    ## simulate
    data = sim.compute_main(N, (v0, tau, k, eps), boundary, T = nbStep, initialization = (init_cond.pos, init_cond.angles), dt = dt)[0]

    return data


def get_mult_data(params: Parameters_Simulation,
                  nb_sim: int):
    
    data = np.zeros((nb_sim, params.nbStep, params.N, 2))

    for i in range(nb_sim):
        data[i, :, :, :] = get_data(params)

    return data



def sims2Graphs(sims: np.array):

    listGraphs = []
    for i_sim in range(sims.shape[0]):
        x, y, attr, inds = ft.processSimulation(sims[i_sim])
        # process it

        for i_t in range(len(x)):
            g = Data(x = x[i_t][:, 2:], y = y[i_t], edge_attr = attr[i_t], edge_index = inds[i_t])
            g = normalizeGraph(g)
            listGraphs.append(g)

    return listGraphs