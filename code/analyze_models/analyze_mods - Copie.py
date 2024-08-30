import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import shutil

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

import dataLoading as dl
from measure import plotStdMessage
import features as ft


from utils.testing_gen import get_mult_data, Parameters_Simulation


NB_SIM = 5
NB_STEP = 1000
MODEL = 'simplest'


##############################
# paths 

#PATH = 'master/code/runs1'
#PATH = 'master/code/runs2'
#PATH = ['/master/code/analyze_models/exps/test_new_activation_0']
#PATH = ['/master/code/analyze_models/exps/exp-test']
PATH = ['/master/code/analyze_models/exps/models-big-simplest']

#DISPLAY_PATH = 'master/code/display_l1'
#DISPLAY_PATH = '/master/code/display_l1_2'
#DISPLAY_PATH = ['/master/code/analyze_models/display/test_new_activation_0']
DISPLAY_PATH = ['/master/code/analyze_models/display/exp-test/new']

MODEL_PATH = '/master/code/models/mod_base'


##############################



def getData(params: Parameters_Simulation, nb:int):
    data = get_mult_data(params, nb)


    dataList = []

    for i in range(data.shape[0]):
        x, y, attr, inds = ft.processSimulation(data[i])

        for i in range(len(x)):
            g = Data(x = x[i][:, 2:], y = y[i], edge_attr = attr[i], edge_index = inds[i])
            dataList.append(g)

        return dataList


def loadModel(modelName:str, inputShape:int = 8, edges_shape = 5, path = None):
    """ 
    Function to import the model

    Args:
    -----
        - `modelName`: name of the model
        - `inputShape`: inout shape of the NN
        - `edges_shape`: edge shape of the NN
        - `path`: path where the models are
    """

    sys.path.append(path)

    loadFun = __import__(f'{modelName}', fromlist = ('loadNetwork'))

    model = loadFun.loadNetwork(inputShape, edges_shape)

    return model


def getName(path):
    run_name = path.split('/')[-3]

    model_type = path.split('/')[-1].split('.')[0]

    if 'best' in model_type:
        model_type = '_best'

    else:
         model_type = '_latest'

    name = run_name + model_type

    return name


##############################
# Messages
#TODO linear 


def findIndices(message, nb = 5):
    stdv=plotStdMessage(message)
    plt.close()

    inds = np.argsort(stdv)
    # change of the order
    return np.flip(inds[-nb:])




def plotMessage(graph, messages, i_attr, id_message):
    edges = graph.edge_attr.cpu().detach().numpy().copy()
    
    plt.scatter(edges[:, i_attr], messages[:, id_message])
    #plt.plot(x, mean, 'green')
    #plt.fill_between(x, mean-std, mean+ std, color = 'red', alpha = 0.4)
    #plt.vlines(x = 2, ymin = np.min(messages[:, id]), ymax = np.max(messages[:, id]))
    #plt.vlines(x = 4, ymin = np.min(messages[:, id]), ymax = np.max(messages[:, id]))


def plotMessageEvol(modelList, pathPlot = DISPLAY_PATH):

    # create folders

    out_poss = ['distance', 'cosine', 'sine', 'radius_1', 'radius_2']
    number_messages = 5
    
    nb = 0
    nb_sim = 0
    with torch.no_grad():
        # get the data for the different models

        for model_path in tqdm(modelList):
            nb_sim += 1

            # path for the experiment
            p_exp = os.path.join(pathPlot, f'exp_{nb_sim}')
            if not os.path.exists(p_exp):
                os.makedirs(p_exp)
            else:
                print('WARNING: weird stuff here')

            for f in out_poss:
                p = os.path.join(p_exp, f)
                if not os.path.exists(p):
                    os.makedirs(p)

            # find name that identify the model
            name_plot = getName(model_path)

            print(name_plot)
            print(model_path)

            # load model
            model = loadModel(MODEL, path=MODEL_PATH)
            
            std_dict = torch.load(model_path)
            model.load_state_dict(std_dict)
            model.eval()

            # get messages
            message = model.message(data).cpu().detach().numpy()



            # best messages indices
            inds = findIndices(message, nb = number_messages)

            plotStdMessage(message.copy())
            p_std = os.path.join(p_exp, f'{name_plot}.png')
            if os.path.exists(p_std):
                print("WARNING >>> issue here")
            plt.savefig(p_std)
            plt.close()

            for i in range(len(out_poss)):      # radius, ...
                for j in range(number_messages):    # id of the message

                    plotMessage(data, message.copy(), i, inds[j])
                    
                    path = os.path.join(p_exp, out_poss[i])
                    path = os.path.join(path, f"{name_plot}_attr-{out_poss[i]}_nb-{j}.png")

                    nb_plot = 0
                    while os.path.exists(path):
                        nb_plot += 1
                        path = os.path.join(p_exp, out_poss[i])
                        path = os.path.join(path, f"{name_plot}_attr-{out_poss[i]}_nb-{j}_nbPlot-{nb_plot}.png")

                    plt.savefig(path)
                    plt.close()

##############################
# linear regression analyzis
#TODO


##############################
# effect on the degree + mean distance + speed on the error
#TODO


## effect of degree


## effect of mean distance


## effect of speed


## effet of continuity


##############################
# overall comp in a folder
#TODO




##############################
# main functions



def main():

    if PATH is None:
        # check if listdir only outputs the last element (...)
        list_exp = [os.listdir('/master/code/analyze_models/exp/')]
        list_disp = [os.path.join('/master/code/analyze_models/display', list_exp[i]) for i in range(len(list_exp))]

    else:
        list_exp = PATH
        list_disp = DISPLAY_PATH


    for i in range(len(list_exp)):

        exp = list_exp[i]
        disp = list_disp[i]

        model_list = findModels(exp)



        # messages analyzis

        plotMessageEvol(model_list, pathPlot = disp)

        # ...



if __name__ == '__main__':
    main()