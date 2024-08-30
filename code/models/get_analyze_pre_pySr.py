import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_regression
from torch_geometric.data import Data




def path_link(path:str):
    sys.path.append(path)

path_link('/home/jpierre/v2/lib')

from features import processSimulation
from norm import normalizeGraph

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATH_MODELS = '/home/jpierre/v2/new_runs/Alan'
PATH_OUTPUT = '/home/jpierre/v2/pySr_pre/0_001'
DATA_PATH = '/scratch/users/jpierre/mew/test/np_file/simulation_0.npy'

def create_fold(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collect_models(path, debug = True):
    modelList = []
    pathList = []


    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pt') or file.endswith('.ptt'):
                if debug:
                    print(f' file >> {file} \n root >>> {root} \n ----------------------\n\n\n')
                modelList.append(os.path.join(root, file))

                pathList.append(root.split('/')[-2])

    return modelList, pathList


def chooseModel(name):
    nameBase = 'simplest'

    if 'no-drop' in name:
        dropout = False
        nameBase += 'no_dropout'
    else:
        dropout = True
        nameBase += '_dropout'

    if 'no-enc' in name:
        encoder = False
        nameBase += '_no-encoder'
    else:
        encoder = True
        nameBase += '_encoder'

    return nameBase

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


def getData(path):
    data =  np.load(path)
    x, y, attr, inds = processSimulation(data)

    return x, y, attr, inds



def plotStd(messages, path, xlabel, ylabel, display:bool = True):

    std = np.std(messages, axis = 0)

    if display:
        plt.plot(std)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(path)
        plt.close()

    return std


def mutual_information_matrix(input, outputs, path, xlabel, ylabel, display:bool = True):
    """
    Compute the mutual information matrix for k vectors.
    
    Args:
    vectors (list of np.array): List containing k vectors.
    
    Returns:
    np.array: A k x k matrix with mutual information scores.
    """

    nb_inputs = input.shape[0]
    nb_outputs = outputs.shape[0]

    mi_matrix = np.zeros((nb_inputs, nb_outputs))
    
    for j in range(nb_outputs):
        mi_matrix[:, j] = mutual_info_regression(input.T, outputs[j, :])
    
    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(mi_matrix, cmap='coolwarm', vmin=0, vmax=np.max(mi_matrix))
        plt.colorbar()

        # Add labels and title
        plt.title('Correlation Matrix Heatmap')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(path)
        plt.close()

    return mi_matrix


def getOrderedVals(attribute, message, bins):
    bin_edges = np.linspace(np.min(attribute), np.max(attribute), bins + 1)
    bin_indices = np.digitize(attribute, bin_edges) - 1

    means = np.zeros(bins)
    stds = np.zeros(bins)

    for i in range(bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            means[i] = np.mean(message[bin_mask])
            stds[i] = np.std(message[bin_mask])
        else:
            means[i] = np.nan
            stds[i] = np.nan

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    return bin_centers, means, stds



def getPlots(input, output, path, xlabel, ylabel):
    for i in range(input.shape[1]):
        for j in range(output.shape[1]):
            p = f'{path}_input-{i}_output-{j}.png'
            x, mean, std = getOrderedVals(input[:, i], output[:, j], 100)

            plt.scatter(input[:, i], output[:, j])
            plt.plot(x, mean, 'green')
            plt.fill_between(x, mean-std, mean+ std, color = 'red', alpha = 0.4)
            plt.xlabel(f'{xlabel}-{i}')
            plt.ylabel(f'{ylabel}-{j}')

            plt.savefig(p)
            plt.close()



def getMessage(mod, graph):
    attr = graph.edge_attr
    vect = mod.getMessage(None, None, attr).cpu().numpy()
    return vect

def getEncoding(mod, graph):
    input = graph.x
    vect = mod.getEncoding(input).cpu().numpy()
    return vect

def getUpdates(mod, graph):
    
    vect = mod(graph).cpu().numpy()
    return vect

def getElements(mod, input, attr):
    message = getMessage(mod)

    enc = getEncoding(mod, input)

    updates = getUpdates(mod, input, attr)

    return message, enc, updates

def getValues(mod, x, y, attr, inds):


    vect = None

    encoding = None
    updates = None

    #x, y, attr, inds = processSimulation(data)
    with torch.no_grad():
        for i in tqdm(range(len(x))):
            graph = Data(x = x[0], edge_index = inds[0], edge_attr = attr[0])
            graph = normalizeGraph(graph)
            graph = graph.to(DEVICE)


            if vect is None:
                vect = getMessage(mod, graph.clone())
            else:
                vect = np.vstack((vect, getMessage(mod, graph.clone())))


            if encoding is None:
                encoding = getEncoding(mod, graph.clone())
            else:
                encoding = np.vstack((encoding, getEncoding(mod, graph.clone())))


            if updates is None:
                updates = getUpdates(mod, graph.clone())
            else:
                updates = np.vstack((updates, getUpdates(mod, graph.clone())))




def createFigures(path, x, y, attr, message, encoding, updates):


    # std figs

    xlabel = 'Message'
    ylabel = 'Std'
    
    if message:
        p = os.path.join(path, 'Message-std.png')
        plotStd(message, p, xlabel, ylabel)

    if encoding:
        xlabel = 'Encoding'
        p = os.path.join(path, 'Encoding-std.png')
        plotStd(encoding, p, xlabel, ylabel)

    if updates:
        xlabel = 'Updates'
        p = os.path.join(path, 'Updates-std.png')
        plotStd(updates, p, xlabel, ylabel)


    # MI figs
    if message:
        p = os.path.join(path, 'Messages-MI.png')
        xlabel = 'Message'
        ylabel = 'Attribute'
        mutual_information_matrix(message, attr, path, xlabel, ylabel)

    if encoding:
        p = os.path.join(path, 'Encoding-MI.png')
        xlabel = 'Encoding'
        ylabel = 'Input'
        mutual_information_matrix(encoding, x, path, xlabel, ylabel)


    #if updates:
    #    p = os.path.join(path, 'Updates-MI.png')
    #    xlabel = 'output'
    #    ylabel = 'updates'
    #    mutual_information_matrix(input, outputs, path, xlabel, ylabel)


    # display

    if message:
        p = os.path.join(path, 'Messages-graphs.png')
        ylabel = 'Message'
        xlabel = 'Attribute'
        getPlots(attr, message, path, xlabel, ylabel)

    if encoding:
        p = os.path.join(path, 'Encoding-graphs.png')
        ylabel = 'Encoding'
        xlabel = 'Input'
        getPlots(x, encoding, path, xlabel, ylabel)


    return None



def main():

    cwd = os.getcwd()
    # 1: get the models and the paths

    modelList, pathList = collect_models(PATH_MODELS)

    # 2: get the data

    x, y, attr, inds = getData(DATA_PATH)
    x_np = None
    y_np = None
    attr_np = None

    # 3: save x, y, attr

    for i in range(len(x)):
        if x_np is None:
            x_np = x[i].numpy()
        else:
            x_np = np.vstack((x_np, x[i].numpy()))

        if y_np is None:
            y_np = y[i].numpy()
        else:
            y_np = np.vstack((y_np, y[i].numpy())) 

        if attr_np is None:
            attr_np = attr[i].numpy()
        else:
            attr_np = np.vstack((attr_np, attr[i].numpy())) 


    np.save(os.path.join(PATH_OUTPUT, 'x.npy'), x_np)
    np.save(os.path.join(PATH_OUTPUT, 'y.npy'), y_np)
    np.save(os.path.join(PATH_OUTPUT, 'attr.npy'), attr_np)

    for i in tqdm(range(len(modelList))):

        # 4: create associated folder
        o_path = os.path.join(cwd, pathList[i])
        create_fold(o_path)

        # 5: get the model (encoding), message, updates
        mod = loadModel(f'{chooseModel({modelList[i]})}_pySr', path = '/home/jpierre/v2/models')
        std_dict = torch.load(modelList[i])
        mod.load_state_dict(std_dict)
        mod.eval()
        mod = mod.to(DEVICE)

        message, encoding, updates = getValues(mod, x, y, attr, inds)

        # 6: save encodings, message

        if message:
            np.save(os.path.join(o_path, 'message.npy'), message)

        if encoding:
            np.save(os.path.join(o_path, 'encoding.npy'), encoding)

        if updates:
            np.save(os.path.join(o_path, 'updates.npy'), updates)


        # 7: create the associated figures

        createFigures(o_path, x_np.copy(), y_np.copy(), attr_np.copy(), message.copy(), encoding.copy(), updates.copy())



if __name__ == '__main__':
    main()