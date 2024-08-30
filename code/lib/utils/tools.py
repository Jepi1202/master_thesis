import numpy as np
import json
import os



def array2List(mat: np.array) -> list:
    res = []

    for i in range(mat.shape[0]):
        res.append(mat[i])


    return res


def makedirs(file):
    f = file

    i = 1
    while os.path.exists(f):
        f = f'{file}_{i}'
        i += 1
    
    os.makedirs(f)

    return f



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
