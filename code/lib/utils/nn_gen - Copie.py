import torch
import numpy as np
from torch_geometric.data import Data
import sys

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

from norm import normalizeGraph
import features as ft
from NNSimulator_dt import genSim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def generate_sim(model, 
                 gt: np.array, 
                 initId:int = 8,
                 nbStep:int = 100,
                 device = DEVICE):
    
    nbStep = nbStep - 1 

    ## get data (gt)
    x, y, attr, inds = ft.processSimulation(gt)
    graphs = []

    for i_t in range(len(x)):
        g = Data(x = x[i_t][:, 2:], y = y[i_t], edge_attr = attr[i_t], edge_index = inds[i_t])
        g = normalizeGraph(g).to(DEVICE)
        graphs.append(g)

    ## get init state
    init_state = graphs[initId]
    pos = x[initId][:, :2]

    ## rollout simulation
    with torch.no_grad():
        out = genSim(model, nbStep, init_state, pos, radius=None, train = False, debug = None)

    return out




def generate_sim_batch(model, 
                        gt: np.array, 
                        initId:int = 8,
                        nbStep:int = 100,
                        device = DEVICE):
    
    res = np.zeros((gt.shape[0], nbStep, gt.shape[-2], 2))

    for i in range(gt.shape[0]):
        res[i] = generate_sim(model, 
                              gt[i], 
                              initId, 
                              nbStep, 
                              device).detach().cpu().numpy()

    return res