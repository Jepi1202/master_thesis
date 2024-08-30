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
from utils.pysr_help import getEdges, get_weights, get_sum_gam_weights, getPrediction
from utils.loading import loadModel
from utils.tools import makedirs




NB_SIM = 2
NB_STEP = 300
PATH_MODELS = '/master/code/experimental/pysr/gam/models/128'
# PATH_MODELS = '/master/code/experimental/pysr/gam/models/GAM_2_20-08/2'
# PATH_MODELS = '/master/code/experimental/pysr/gam/models/GAM_2_20-08/128'



DISPLAY_PATH = '/master/code/experimental/pysr/gam/weights_evol/figures_test'
MODEL = 'GAM_GNN_128'
#MODEL = 'GAM_GNN_2'
#MODEL = 'GAM_GNN-relu'




def directScat(radius, vals, name, save_path = None):
    plt.scatter(radius, vals)

    plt.xlabel('Radius')
    plt.ylabel(name)

    if save_path:
        plt.savefig(save_path)
        plt.close()




def scatPlot(weights, sum_weights, radius, names, path, display = False, degree = 2):
    
    
    for i in range(len(names)):
        plt.scatter(radius, weights[:, i])
        plt.xlabel("Radius [micro m]")
        plt.ylabel(f"Weight times {names[i]}")
        plt.tight_layout()

        name_plot = os.path.join(path, f'{names[i]}.png')
        k = 0
        while os.path.exists(name_plot):
            name_plot = f'{name_plot.split(".")[0]}-{k}.png'
            k+=1


        if display: 
            plt.show()
        else:
            plt.savefig(name_plot)
            plt.close()
    
    # sum plot
    
    for i in range(degree):
        
        plt.scatter(radius,sum_weights)
        plt.tight_layout()

        # PREVIOUS                          plt.scatter(radius,sum_weights[:, i])
        plt.xlabel("Radius [micro m]")
        plt.ylabel(f"Sum Weight - {i}")

        name_plot = os.path.join(path, f'sum-weights-{i}.png')

        k = 0
        while os.path.exists(name_plot):
            name_plot = f'{name_plot.split(".")[0]}-{k}.png'
            k+=1

        if display: 
            plt.show()
        else:
            plt.savefig(name_plot)
            plt.close()


def plotStd(message, path):
    plt.plot(np.std(message, axis = 0))
    plt.savefig(path)
    plt.close()


    

def compute_num_products(num_variables, degree):
    """
    Compute the number of polynomial products for a given number of variables and degree.
    
    Args:
        num_variables (int): The number of variables.
        degree (int): The maximum degree of the polynomial terms.

    Returns:
        int: The total number of polynomial products.
    """
    num_products = 0
    for d in range(1, degree + 1):
        num_products += sum(1 for _ in itertools.combinations_with_replacement(range(num_variables), d))
    return num_products


def compute_products_batch(tensor, degree):
    """
    Compute all polynomial products up to a given degree for a batch of input tensors using PyTorch operations.
    
    Args:
        tensor (torch.Tensor): The input tensor with shape [batch_size, num_variables].
        degree (int): The maximum degree of the polynomial terms.

    Returns:
        torch.Tensor: A tensor containing all polynomial products for the batch, shape [batch_size, num_products].
    """
    batch_size, num_variables = tensor.shape
    num_products = compute_num_products(num_variables, degree)
    products = torch.empty(batch_size, num_products, device=tensor.device)

    idx = 0
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(num_variables), d):
            product = torch.prod(tensor[:, list(combo)], dim=1)
            products[:, idx] = product
            idx += 1

    return products


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

                out_list.append(os.path.join(display_path, f'test-{nb}-{name}'))
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


def findIndices(vect, nb = 5):
    

    inds = np.argsort(vect)
    # change of the order
    return np.flip(inds[-nb:])

def main():
    """
    Display the weights of the GAM
    """

    # find the models

    models, path_out_models = find_models_and_paths(PATH_MODELS, DISPLAY_PATH)

    print(models)
    print(path_out_models)

    # get the data

    params = Parameters_Simulation()
    params.nbStep = NB_STEP
    dataList = getData(params, nb = NB_SIM)
    edges = getEdges(dataList)

    variables = [' r ', ' cos ', ' sin ', ' R1 ', ' R2 ']
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
        print(path)
        makedirs(path)

        weights, edgs = get_weights(mod, dataList, nbMax = None)
        sum_w = get_sum_gam_weights(mod, dataList)
        #med_weights = np.median(weights, axis = 0)

        # std messages

        messages = getPrediction(mod, dataList)
        std_messages = np.std(messages, axis = 0)



        plt.bar(np.arange(std_messages.shape[0]), std_messages)
        plt.ylabel('Std of the messages')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'messages_std.png'))

        plt.close()

        #plotStd(messages, os.path.join(path, 'messages_std.png'))

        inds_messages = findIndices(std_messages)
        print(inds_messages)


        plotStd(weights, os.path.join(path, 'weights_std.png'))
        std_w = np.std(weights, axis = 0)

        inds_w = findIndices(std_w)



        median_w = np.median(np.abs(weights), axis = 0)
        plt.plot(median_w)
        plt.xlabel('Weights')
        plt.ylabel('Median of the values')
        plt.savefig(os.path.join(path, 'weights_median.png'))
        plt.close()

        # get the names
        w2 = weights.copy().reshape(weights.shape[0],-1,  20)
        nb_messages = w2.shape[1]
        names_full = result * nb_messages

        print(f'Inds w -- {inds_w.shape}')

        # plot only for the best weights

        for j in range(len(inds_w)):
            w_j = inds_w[j]

            
            path_w = os.path.join(path, f'weights-{w_j}-{j}.png')
            #os.makedirs(path_w)

            directScat(vals = weights[:, w_j], 
                    radius = edgs[:, 0], 
                    name = names_full[w_j], 
                    save_path = path_w)
            

        
            
        

        # for the best messages, all

        for j in range(len(inds_messages)):
            print(j)

            print(f'Inds w -- {inds_messages.shape}')


            mess_j = inds_messages[j]
            print(mess_j)


            w_mess = w2[:, mess_j, :]
            print(w_mess.shape)
            sum_w_mess = sum_w[:, mess_j]
            

            path_messages = os.path.join(path, f'messages-{mess_j}-{j}')
            os.makedirs(path_messages)
            print(path_messages)



            ####
            p_r = os.path.join(path_messages, 'rad')
            os.makedirs(p_r)

            scatPlot(w_mess, 
                    radius = edgs[:, 0],
                    sum_weights=sum_w_mess, 
                    names = result, 
                    path = p_r, 
                    display = False,
                    degree = 1)
            

            ####
            p_cos = os.path.join(path_messages, 'cos')
            os.makedirs(p_cos)

            scatPlot(w_mess, 
                    radius = edgs[:, 1],
                    sum_weights=sum_w_mess, 
                    names = result, 
                    path = p_cos, 
                    display = False,
                    degree = 1)
            

            ####
            p_sin = os.path.join(path_messages, 'sin')
            os.makedirs(p_sin)

            scatPlot(w_mess, 
                    radius = edgs[:, 2],
                    sum_weights=sum_w_mess, 
                    names = result, 
                    path = p_sin, 
                    display = False,
                    degree = 1)
            

            ####
            p_rad1 = os.path.join(path_messages, 'rad1')
            os.makedirs(p_rad1)

            scatPlot(w_mess, 
                    radius = edgs[:, 3],
                    sum_weights=sum_w_mess, 
                    names = result, 
                    path = p_rad1, 
                    display = False,
                    degree = 1)
            

            ####
            p_rad2 = os.path.join(path_messages, 'rad2')
            os.makedirs(p_rad2)

            scatPlot(w_mess, 
                    radius = edgs[:, 4],
                    sum_weights=sum_w_mess, 
                    names = result, 
                    path = p_rad2, 
                    display = False,
                    degree = 1)
            



            # plot bar of median

            median_weights_message = np.median(np.abs(w_mess), axis= 0)
            plt.bar(result, median_weights_message)
            plt.ylabel('Median of the weights')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'messages-{mess_j}-{j}/meds.png'))

            plt.close()



            median_weights_message = np.std(w_mess, axis= 0)
            plt.bar(result, median_weights_message)
            plt.ylabel('Std of the weights')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'messages-{mess_j}-{j}/stds.png'))

            plt.close()









        """

        print(weights.shape)
        for j in range(len(inds_messages)):

            w_mess = weights[]

        
            # function to display

            print(weights.shape)
            scatPlot(weights, 
                    radius = edgs[:, 0],
                    sum_weights=sum_w, 
                    names = result * 2, 
                    path = path, 
                    display = False,
                    degree = degree)

        """
        

        


if __name__ == '__main__':
    main()

        
