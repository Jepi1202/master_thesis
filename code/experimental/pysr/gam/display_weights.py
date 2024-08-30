import os
import numpy as np
import sys
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import itertools



def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')


from utils.testing_gen import get_mult_data, Parameters_Simulation
import features as ft
from utils.pysr_help import getEdges, get_weights
from utils.loading import loadModel
from utils.tools import makedirs



NB_SIM = 5
NB_STEP = 1000
PATH_MODELS = '/master/code/experimental/pysr/gam/models/exp_1_relu'
DISPLAY_PATH = '/master/code/experimental/pysr/gam/Weights'
#MODEL = 'GAM_GNN'
MODEL = 'GAM_GNN-relu'


def pcolorPlot(vals, labels, num_parts):
    vals_reshaped = vals.reshape(1, -1)
    total_length = vals_reshaped.shape[1]
    part_size = total_length // num_parts

    fig, axes = plt.subplots(num_parts, 1, figsize=(6, 2.5 * num_parts), gridspec_kw={'height_ratios': [1]*num_parts, 'hspace': 0})
    
    if num_parts == 1:
        axes = [axes]
    
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = start_idx + part_size if i < num_parts - 1 else total_length
        
        vals_part = vals_reshaped[:, start_idx:end_idx]
        labels_part = labels[start_idx:end_idx]
        
        ax = axes[i]
        c = ax.pcolormesh(vals_part, cmap='gray_r', edgecolors='k')
        
        if i == 0: 
            for j, label in enumerate(labels_part):
                ax.text(j + 0.5, 1.25, label, ha='center', va='bottom', fontsize=10, rotation=90, color='black')
        
        ax.axis('off')
        ax.set_aspect('equal')

    plt.subplots_adjust(hspace=0.01, top=0.50, bottom=0.20)


def getData(params: Parameters_Simulation, nb:int):
    data = get_mult_data(params, nb)


    dataList = []

    for i in range(data.shape[0]):
        x, y, attr, inds = ft.processSimulation(data[i])

        for i in range(len(x)):
            g = Data(x = x[i][:, 2:], y = y[i], edge_attr = attr[i], edge_index = inds[i])
            dataList.append(g)

        return dataList
    
    


def find_models_and_paths(model_path:str, display_path:str)->tuple:
    model_list = []
    out_list = []
    
    nb = 0

    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('pt'):
                model_list.append(os.path.join(root, file))

                name = file.split('.')[0]

                out_list.append(os.path.join(display_path, f'test-{nb}-{name}.png'))
                nb += 1


    return model_list, out_list


def compute_string_products(strings, degree):
    """
    Compute all polynomial products up to a given degree for a list of strings.
    
    Args:
        strings (list of str): The input list of strings.
        degree (int): The maximum degree of the polynomial terms.

    Returns:
        list of str: A list containing all polynomial products.
    """
    num_variables = len(strings)
    products = []

    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(num_variables), d):
            # Create a product string by joining the corresponding strings
            product = ''.join([strings[i] for i in combo])
            products.append(product)

    return products

def main():
    """
    Display the weights of the GAM
    """

    # find the models

    models, path_out_models = find_models_and_paths(PATH_MODELS, DISPLAY_PATH)

    # get the data

    params = Parameters_Simulation()
    params.nbStep = NB_STEP
    dataList = getData(params, nb = NB_SIM)
    edges = getEdges(dataList)

    variables = [r'$r \;$', r'$\cos(\theta) \;$', r'$\sin(\theta) \;$', r'$R1 \;$', r'$R2 \;$']
    degree = 2
    result = compute_string_products(variables, degree)


    # get the 

    for i in tqdm(range(len(models))):
        model = models[i]
        mod = loadModel(MODEL)
        std_dict = torch.load(models[i], map_location="cpu")
        mod.load_state_dict(std_dict)
        mod.eval()

        path = path_out_models[i]

        weights, edgs = get_weights(mod, dataList, nbMax = None)
        med_weights = np.median(weights, axis = 0)
        
        # function to display

        pcolorPlot(np.median(weights, axis = 0), num_parts=2, labels = result)
        
        plt.savefig(path)
        plt.close()


if __name__ == '__main__':
    main()

        
