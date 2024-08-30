from pysr import PySRRegressor
import sys
import os
import yaml
import numpy as np

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

from utils.tools import writeJson, readJson



PATH = '/home/jpierre/v2/pySr/data/mod1/data.json'


with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 


NB_RUN = cfg['NB_RUN']
BINARY_OP = ["+", "*"]
UNARY_OP = ["inv(x) = 1/x"]
VARIABLES = ['v_x_0', 'v_y_0', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'd', 'delta_x', 'delta_y', 'r_i', 'r_j']
MAX_SIZE = cfg['MAX_SIZE']
PARSIMONY = cfg['PARSIMONY']
X_ELEM = cfg['X_ELEM']
LIM = cfg['LIM']


KEY = cfg['KEY']



def getData(jsonPath = PATH, key = None):
    # for now


    data = readJson(jsonPath)

    if X_ELEM == 'edges': 

        if key is None:
            X = data['edges']
            Y = data['message']

        else:
            X = data[key]['edges']
            Y = data[key]['message']

    if X_ELEM == 'out':
        if key is None:
            X = data['sum_message']
            Y = data['out']

        else:
            X = data[key]['sum_message']
            Y = data[key]['out']
    
    return X, Y



def getPySrModel(nbRun = NB_RUN, binaryOp = BINARY_OP, unaryOp = UNARY_OP, maxsize = MAX_SIZE):

    model = PySRRegressor(
        niterations=nbRun,
        binary_operators=binaryOp,
        #unary_operators= ["inv(x) = 1/x",],
        #populations=15,
        model_selection = "best",
        maxsize = maxsize,
        #complexity_of_variables = 2,
        parsimony = PARSIMONY,
        #nested_constraints = {"cond": {"*":0}},
        #adaptive_parsimony_scaling = 1000,
        #ncycles_per_iteration = 1000,
        #turbo = True,
        #extra_sympy_mappings = {"inv": lambda x: 1 / x},
        elementwise_loss = 'L1DistLoss()'
    )

    return model


def fittingModel(model, X, y, verbose:bool = False, variables = VARIABLES):

    if verbose:
        print(">>>>> Fitting pySr")


    # Fit model
    model.fit(X, y, 
    variable_names = variables,
    # X_units = [""],
    # y_units = "",
    )


    return model



def main():

    X, Y = getData(PATH, KEY)


    if LIM != -1:
        X = X[:LIM, :]
        Y = Y[:LIM, :]



    pyReg = getPySrModel()


    if X_ELEM == 'edges':
        mod = fittingModel(pyReg, X.copy()[:, :3], Y.copy(), variables=['r', 'r_x', 'r_y'])
    elif X_ELEM == 'out':
        mod = fittingModel(pyReg, X, Y.copy(), variables=None)


    #cwd = os.getcwd()
    #p = os.path.join(cwd, f'table_{KEY}.tex')


    # does not work for now :(
    print(mod.latex_table(()))



if __name__ == '__main__':
    main()