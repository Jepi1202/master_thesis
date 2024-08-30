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
from utils.loading import loadModel
import wandb



#DEVICE = 'cuda:0'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH_BASE = '/home/jpierre/v2/path_datasets'

def getLoader(path, batch_size = 32, shuffleBool = True, root = None, jsonFile = None, mode = 'training'):
    datasetTraining = dl.DataLoader2(root, path = path, jsonFile = jsonFile, mode = mode)
    loader = DataLoader(datasetTraining, batch_size=batch_size, shuffle = shuffleBool)
    
    return loader


def getDict(modelName:str, cfg:dict):

    if 'simplest' in modelName:
        return cfg['simplest']
    
    elif 'gat' in modelName:
        return cfg['gat']
    
    elif 'GAM' in modelName:
        return cfg['GAM']
    
    elif 'complex' in modelName:
        return cfg['complex']
    




def main(device = DEVICE):
    root = os.getcwd()
    params = tr.paramsLearning()
    
    with open('cfg.yml', 'r') as file:
        cfg = yaml.safe_load(file) 
    trainingInfos = cfg['training']
    modelInfo = cfg['Model']

    ##########
    ## Main elemets

    params.nbEpoch = trainingInfos['nbEpoch']
    params.batchSize = trainingInfos['batch']
    device_tr = trainingInfos['device']


    ##########
    ## Model elemets
    params.modelName = modelInfo['modelName']
    params.model_dict = getDict(params.modelName, modelInfo)
    model = loadModel(params.modelName, params.model_dict)
    model = model.to(device)

    ##########
    ## Evaluate
    
    params.freqEval = trainingInfos['frequenceEval']
    params.minEvalLoss = float('inf')
    batchSizeEval =  trainingInfos['batchSizeVal']


    ##########
    ## Loss elemets

    params.type_training = trainingInfos['Type']['type']

    if params.type_training == 'one-step':
        params.nbRoll = None
    else:
        params.nbRoll = trainingInfos['Type']['rolloutNb']


    params.L1LossReg = trainingInfos['loss']['l1Reg']
    params.lossScaling = trainingInfos['loss']['lossScaling']

    topk = trainingInfos['loss']['topk']
    if topk == -1:
        params.topk = None
    else:
        params.topk = topk

    ##########
    ## Data elemets

    p_data = trainingInfos['Data']['pathData']         # scratch path to the data
    pathJsonBool = trainingInfos['Data']['pathJsonBool']        # bool if json available (true by default)
    
    params.pathData = p_data
    p_training = os.path.join(p_data, 'training/torch_file')
    p_sim = os.path.join(p_data, 'validation/np_file')
    p_json = os.path.join(PATH_BASE, f'{p_data.split("/")[-1]}.json')
    print(f'TEST new path >>> {p_json}')
    
    if pathJsonBool:
        loaderTraining = getLoader(p_data, batch_size = params.batchSize, jsonFile = p_json, mode = 'training')

        loaderEval = getLoader(p_data, batch_size = batchSizeEval, jsonFile = p_json, mode = 'validation')
    else:
        loaderTraining = getLoader(p_data, batch_size = params.batchSize, mode = 'training')

        loaderEval = getLoader(p_data, batch_size = params.batchSize, mode = 'validation')
    

    loaderSim = dl.simLoader2(p_sim)   # possible issue here (update the saving of files)
    loaderSim = DataLoaderPy(loaderSim, batch_size=1, shuffle=False)
    
    
    params.loader = loaderTraining
    params.evalLoaderTorch = loaderEval
    params.evalLoaderSim = loaderSim


    ##########
    ## Optimizer elemets

    params.optimizer_type = trainingInfos['Optimizer']['type']
    params.lr = trainingInfos['Optimizer']['lr']
    params.wdecay = trainingInfos['Optimizer']['lambdaL2Weights']

    params.optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay= params.wdecay)

    ##########
    ## Scheduler elemets
    

    schedule_type = trainingInfos['Scheduler']['scheduler']
    if schedule_type == 'exp':
        schedule_params = trainingInfos['Scheduler']['exp']['gamma']
    elif schedule_type == 'linear':
        pass
    
    schedule = tr.PersScheduler(params.optimizer, schedule_type, params=schedule_params)
    params.schedule = schedule



    ##########
    ## Data augmentation

    params.augBool = trainingInfos['dataAugment']['bool']
    params.probDataAug = trainingInfos['dataAugment']['prob']
    params.stdSpeed = trainingInfos['dataAugment']['stdSpeed']
    params.stdDeltaPos = trainingInfos['dataAugment']['stdDeltaPos']

    ##########
    ## Save

    params.freqSave = trainingInfos['Save']['frequenceSave']
    params.wbName = trainingInfos['Save']['wbName']

    ## Depository where to put models

    depo_trained = os.path.join(root, 'model_trained')
    if not os.path.exists(depo_trained):
        os.makedirs(depo_trained)
        
    params.pathSaveEval = os.path.join(depo_trained, trainingInfos['Save']['evalModel'])
    params.pathSaveFull = os.path.join(depo_trained, trainingInfos['Save']['saveModel'])



    ##########
    # initiate the wb
    
    
    wandb.init(project = 'master_thesis', name = f"{params.wbName}")


    ##########
    # run the training
    
    if params.type_training == 'one-step':
        print('<<<<<< INFO: training on 1 step transitions >>>>>>')
        model = tr.train_1step(model, params, device = device_tr)
    
    else:
        print('<<<<<< INFO: training on rollouts >>>>>>')
        model = tr.train_roll(model, params, device = device_tr)
    
                               

if __name__ == '__main__':
    main()