from pysr import PySRRegressor
import simpy as sp
import os
import yaml
import numpy as np


FOLDER_OUT = 'dep_latex'

# yaml: null = python: None

def load_configuration(path, configName = 'config_pysr.yml'):
    p = os.path.join(path, configName)

    # laod yml file


def getData(path, x_path = 'x_data', y_path = 'y_data'):
    X_p = np.load(os.path.join(path, x_path))
    Y_p = np.load(os.path.join(path, y_path))

    return X_p, Y_p


def saveResult(path, table, outputFile, timeInd = True, folder = FOLDER_OUT):

    if folder:
        folder_path = os.path.join(path, folder)

    # combine outputFile with date + hour indication (while keeping the same extension, probably .tex)

    # save said 


def getPySrModel(d):

    model = PySRRegressor(
        niterations=d['niterations'],
        binary_operators=d['binary_operators'],
        unary_operators= d['unary_operators'],
        populations=d['population'],
        model_selection = "best",
        maxsize = d['maxsize'],
        parsimony = d['parsimony'],
        nested_constraints = d['nested_constraints'],
        adaptive_parsimony_scaling = d['adaptive_parsimony_scaling'],
        ncycles_per_iteration = d['ncycles_per_iteration'],
        turbo = d['turbo'],
        extra_sympy_mappings = d['extra_sympy_mappings']
    )

    return model


def fittingModel(model, X, y, verbose:bool = False, d = None):

    if verbose:
        print(">>>>> Verbose activated")

    
    if d is None:
        model.fit(X, y, )
        return model

    else:
        model.fit(X, y, 
        variable_names = d['variable_names'],
        X_units = d['X_units'],
        y_units = d['y_units'],
        )


    return model




def pipeline(config = 'config.yml', x_name = 'x_data', y_name = 'y_data' ,output_file_name = 'output.tex', timeInd = True):
    cwd = os.getcwd()

    d = load_configuration(cwd, config)

    X, y = getData(cwd, x_name, y_name)

    model = getPySrModel(d['model_hyper'])

    mod = fittingModel(model, X.copy(), y.copy(), d = d['vars'])

    table = mod.latex_table()

    saveResult(cwd, table, output = output_file_name, timeInd = timeInd)