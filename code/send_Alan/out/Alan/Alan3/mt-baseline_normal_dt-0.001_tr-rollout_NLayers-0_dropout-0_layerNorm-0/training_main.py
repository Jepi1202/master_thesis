import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoaderPy
import sys

def path_link(path:str):
    sys.path.append(path)

path_link('/home/jpierre/v2/lib')
    
import dataLoading as dl
import yaml
import os
import training as tr
import wandb



#DEVICE = 'cuda:0'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PATH_BASE = '/home/jpierre/v2/path_datasets'
PATH_MODEL = '/home/jpierre/v2/models'

def getLoader(path, batch_size = 32, shuffleBool = True, root = None, jsonFile = None, mode = 'training'):
    datasetTraining = dl.DataLoader2(root, path = path, jsonFile = jsonFile, mode = mode)
    loader = DataLoader(datasetTraining, batch_size=batch_size, shuffle = shuffleBool)
    
    return loader



def loadModel(modelName:str, d, path = PATH_MODEL):
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

    #model = loadFun.loadNetwork(inputShape, edges_shape)
    model = loadFun.loadNetwork(d)

    return model


def main(device = DEVICE):
    root = os.getcwd()
    params = tr.paramsLearning()
    
    with open('cfg.yml', 'r') as file:
        cfg = yaml.safe_load(file) 



    trainingInfos = cfg['training']

    d_model = trainingInfos['cfg_mod'][f'{trainingInfos["cfg_mod"]["model_name"]}']

    ##########
    ## Main elemets

    params.nbEpoch = trainingInfos['nbEpoch']
    params.lr = trainingInfos['lr']
    params.freqSave = trainingInfos['frequenceSave']
    params.batchSize = trainingInfos['batch']
    params.dt_update = trainingInfos['dt_update']
    training_type = trainingInfos['training_type']
    #params.shuffleBool = True

    ##########
    ## Loss

    params.wdecay = trainingInfos['loss']['lambdaL2Weights']
    params.L1LossReg = trainingInfos['loss']['l1Reg']
    params.lossScaling = trainingInfos['loss']['lossScaling']

    topk = trainingInfos['loss']['topk']
    if topk == -1:
        params.topk = None
    else:
        params.topk = topk



    ##########
    ## Data augmentation

    params.augBool = trainingInfos['dataAugment']['bool']
    params.probDataAug = trainingInfos['dataAugment']['prob']
    params.stdSpeed = trainingInfos['dataAugment']['stdSpeed']
    params.stdDeltaPos = trainingInfos['dataAugment']['stdDeltaPos']

    ##########
    ## Depository where to put models

    depo_trained = os.path.join(root, 'model_trained')
    if not os.path.exists(depo_trained):
        os.makedirs(depo_trained)
        
    params.pathSaveEval = os.path.join(depo_trained, trainingInfos['evalModel'])
    params.pathSaveFull = os.path.join(depo_trained, trainingInfos['saveModel'])

    ##########
    ## Evaluate
    
    params.freqEval = trainingInfos['frequenceEval']
    params.minEvalLoss = float('inf')
    batchSizeEval =  trainingInfos['batchSizeVal']


    ##########
    ## Data loaders

    p_data = trainingInfos['pathData']         # scratch path to the data
    pathJsonBool = trainingInfos['pathJsonBool']        # bool if json available (true by default)
    
    params.pathData = p_data
    p_training = os.path.join(p_data, 'training/torch_file')
    p_sim = os.path.join(p_data, 'validation/np_file')
    p_json = os.path.join(PATH_BASE, f'{p_data.split("/")[-1]}.json')
    print(f'TEST new path >>> {p_json}')
    

    if training_type == '1-step':
        if pathJsonBool:
            loaderTraining = getLoader(p_data, batch_size = params.batchSize, jsonFile = p_json, mode = 'training')

            loaderEval = getLoader(p_data, batch_size = batchSizeEval, jsonFile = p_json, mode = 'validation')
        else:
            loaderTraining = getLoader(p_data, batch_size = params.batchSize, mode = 'training')

            loaderEval = getLoader(p_data, batch_size = params.batchSize, mode = 'validation')

    elif training_type == 'rollout':
        if pathJsonBool:
            loaderTraining = getLoader(p_data, batch_size = trainingInfos['batch_rollout'], jsonFile = p_json, mode = 'training')

            loaderEval = getLoader(p_data, batch_size = batchSizeEval, jsonFile = p_json, mode = 'validation')
        else:
            loaderTraining = getLoader(p_data, batch_size = trainingInfos['batch_rollout'], mode = 'training')

            loaderEval = getLoader(p_data, batch_size = params.batchSize, mode = 'validation')
        

    loaderSim = dl.simLoader2(p_sim)   # possible issue here (update the saving of files)
    loaderSim = DataLoaderPy(loaderSim, batch_size=1, shuffle=False)
    
    
    params.loader = loaderTraining
    params.evalLoaderTorch = loaderEval
    params.evalLoaderSim = loaderSim
    
    ##########
    ## Model

    modelName = trainingInfos['modelName']
    featureShape = cfg['feature']['inShape']
    edgeShape = cfg['feature']['edgeShape']
    model = loadModel(modelName, d_model)
    
    model = model.to(device)

    ##########
    ## Optimizer

    params.optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay= params.wdecay)
    
    ##########
    ## Scheduler

    schedule_type = trainingInfos['scheduler']
    if schedule_type == 'exp':
        schedule_params = trainingInfos['scheduler_exp']['gamma']
    elif schedule_type == 'linear':
        pass
    
    schedule = tr.PersScheduler(params.optimizer, schedule_type, params=schedule_params)
    params.schedule = schedule


    ##########
    # initiate the wb
    
    print(trainingInfos['tag'])
    print(type(trainingInfos['tag']))
    print('-------------------------------------')
    params.wbName = trainingInfos['wbName']
    wandb.init(project = 'master_thesis', name = f"{params.wbName}", tags = trainingInfos['tag'])


    ##########
    # run the training
    
    training_type = trainingInfos['training_type']
    print(training_type)
    print('-------------------------------------------------------')
    
    if training_type == '1-step':
        model = tr.train_1step(model, params, device = device)

    elif training_type == 'rollout':
        model = tr.train_roll(model, params, device = device)
    
                               

if __name__ == '__main__':
    main()