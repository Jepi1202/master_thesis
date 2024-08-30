from tqdm import tqdm
import os
import shutil
import sys


def findModels(path):
    pathLists = []
    for root, dirs, files in tqdm(os.walk(path)):
            for file in files:
                  
                  if file.endswith('.pt'):
                        pathLists.append(os.path.join(root, file))


    return pathLists


def delete_wandb_dirs(start_path):
    for root, dirs, files in os.walk(start_path, topdown=False):
        for dir_name in dirs:
            if dir_name == "wandb":
                dir_path = os.path.join(root, dir_name)
                print(f"Deleting: {dir_path}")
                shutil.rmtree(dir_path)


def getName(path):
    run_name = path.split('/')[-3]

    model_type = path.split('/')[-1].split('.')[0]

    if 'best' in model_type:
        model_type = '_best'

    else:
         model_type = '_latest'

    name = run_name + model_type

    return name



def getModelName(key):

    name = ''

    #if 'simplest' in key:
    #    name = name + 'simplest'
    
    name = name + 'simplest'

    ## other possibilities

    if 'no-dropout' in key:
        name = name + '_no-dropout'
    else:
        name = name + '_dropout'

    if 'no-encoder' in key:
        name = name + '_no-encoder'
    else:
        name = name + '_encoder'

    if 'relu' in key:
        name = name + '-relu'

    return name



def loadModel(modelName:str, d:dict = None, path:str = '/master/code/models'):
    """ 
    Function to import the model

     '/home/jpierre/v2/models'

    Args:
    -----
        - `modelName`: name of the model
        - `inputShape`: inout shape of the NN
        - `edges_shape`: edge shape of the NN
        - `path`: path where the models are
    """

    sys.path.append(path)

    loadFun = __import__(f'{modelName}', fromlist = ('loadNetwork'))

    model = loadFun.loadNetwork(d)

    return model

