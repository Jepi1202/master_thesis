from pysr import PySRRegressor
#import simpy as sp
import os
import yaml
import numpy as np

import json


PATH = '/home/jpierre/v2/pySr/data/mod1/data.json'


NB_RUN = 1000
BINARY_OP = ["+", "*"]
UNARY_OP = ["inv(x) = 1/x"]
VARIABLES = ['v_x_0', 'v_y_0', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3', 'd', 'delta_x', 'delta_y', 'r_i', 'r_j']
MAX_SIZE = 10
PARSIMONY = 0.0000032


KEY = '3-4'


def readJson(filePath:str):
    """
    Function to read json 
    """
    
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data


def writeJson(data, filePath):
    """
    Function to write json 
    """
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=2)


def getData(jsonPath = PATH):
    # for now


    data = readJson(jsonPath)

    return data



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
        #extra_sympy_mappings = {"inv": lambda x: 1 / x}
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
    d = getData()

    for k in d.keys():
    
        d[k]['attr'] = np.array(d[k]['attr'])
        d[k]['nodes'] = np.array(d[k]['nodes'])
        d[k]['messages'] = np.array(d[k]['messages'])


    key = KEY

    X = d[key]['attr'].copy()
    X[:, 1] = X[:, 1] * X[:, 0]
    X[:, 2] = X[:, 2] * X[:, 0]

    Y = d[key]['messages'].copy()


    pyReg = getPySrModel()

    mod = fittingModel(pyReg, X[:, :3], Y, variables=['r', 'r_x', 'r_y'])

    #cwd = os.getcwd()
    #p = os.path.join(cwd, f'table_{KEY}.tex')

    print(mod.sympy())



if __name__ == '__main__':
    main()