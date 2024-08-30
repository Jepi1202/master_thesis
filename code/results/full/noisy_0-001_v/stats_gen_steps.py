import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree as deg


def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

import simulation_v2 as sim
import features as ft
import utils.loading as load
import utils.testing_gen as gen
from utils.tools import array2List, makedirs, writeJson
import utils.nn_gen as nn_gen

import pandas as pd
import matplotlib.ticker as ticker




DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


#PATH = 'master/code/runs1'
#PATH = 'master/code/runs2'
#PATH = ['/master/code/analyze_models/exps/test_new_activation_0']
PATH = ['/master/code/results/models/noisy/0-001/v']

#DISPLAY_PATH = 'master/code/display_l1'
#DISPLAY_PATH = '/master/code/display_l1_2'
#DISPLAY_PATH = ['/master/code/analyze_models/display/test_new_activation_0']
DISPLAY_PATH = [os.path.join(os.getcwd(), 'figures')]

MODEL_PATH = '/master/code/models'

NB_SIM = 5



class ID():
    def __init__(self):
        self.features_x = None
        self.features_edge = None
        self.MLP_hidden = None
        self.l1 = None
        self.dropout = None
        self.layer_norm = None
        self.nb_layer = None
        self.model = None
        self.dt = None

        self.data_type = None


    def get_name(self):
        name = f'{self.model}_'

        if self.data_type:
            name += f'dType-{self.data_type}_'

        if self.features_x:
            name += f'featX-{self.features_x}_'

        if self.features_edge:
            name += f'featE-{self.features_edge}_'

        if self.MLP_hidden:
            name += f'nbHiddenMLP-{self.MLP_hidden}_'

        if self.l1:
            name += f'l1-{self.l1}_'

        if self.dropout:
            name += f'dropout-{self.dropout}_'

        if self.layer_norm:
            name += f'layerNorm-{self.layer_norm}_'

        if self.nb_layer:
            name += f'nbLayer-{self.nb_layer}_'

        if self.dt:
            name += f'dt-{self.dt}_'


        if name.endswith('_'):
            name = name[:-1]

        return name
    

    def load_from_wb(self, name):

        splits = name.split('_')

        self.model = 'simplest'

        if 'baseline' in name:
            self.model = 'baseline'

        if 'complex' in name:
            print(name)
            self.model = 'gnn' 

        if 'gat' in name:
            self.model = 'gat'


        #########

        if 'noisy' in name:
            self.data_type = 'noisy'

        elif 'normal' in name:
            self.data_type = 'normal'

        for s in splits:
            if 'featX' in s:
                self.features_x =s.split('-')[-1]
            if 'featE' in s:
                self.features_edge = s.split('-')[-1]
            if 'nbHiddenMLP' in s:
                self.MLP_hidden = int(s.split('-')[-1])
            if 'l1' in s:
                self.l1 = float(s.split('-')[-1])
            if 'dropout' in s:
                self.dropout = s.split('-')[0]
            if 'layerNorm' in s:
                self.layer_norm = s.split('-')[-1]
            if 'nbLayer' in s:
                self.nb_layer = int(s.split('-')[-1])
            if 'dt' in s:
                self.dt = float(s.split('-')[-1])


            if 'dType' in s:
                self.features_x =s.split('-')[-1]



        



def getParams():
    params = gen.Parameters_Simulation()  


    params.dt = 0.001
    params.v0 = 60
    params.k = 70
    params.epsilon = 0.5
    params.tau = 3.5
    params.R = 1
    params.N = 200
    params.boundary = 100
    params.nbStep = 300


    params.noisy = 1        # function dans utils
    params.features_x = 'v'
    params.features_edge = 'first'


    return params

from utils.stats import perform_1_step_stats

def main():

    if PATH is None:
        # check if listdir only outputs the last element (...)
        list_exp = [os.listdir('/master/code/analyze_models/exp/')]
        list_disp = [os.path.join('/master/code/analyze_models/display', list_exp[i]) for i in range(len(list_exp))]

    else:
        list_exp = PATH
        list_disp = DISPLAY_PATH



    # praamters of the silualtiosn

    params = getParams()                # adapt ...
    data_gt = gen.get_mult_data(params, NB_SIM)
    graphs_gt = gen.sims2Graphs(data_gt, params.features_x)

    print(params.dt)
    
    perform_1_step_stats(data_gt, graphs_gt, list_exp, list_disp, params)


if __name__ == '__main__':
    main()