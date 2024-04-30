import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoaderPy
import sys

def path_link(path:str):
    sys.path.append(path)

path_link('/home/jpierre/v2/lib')
    
import training as tr
import dataLoading as dl
import yaml
import os
import wandb

#DEVICE = 'cuda:0'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getLoader(path, batch_size = 32, shuffleBool = True, root = None, jsonFile = None, mode = 'training'):
    datasetTraining = dl.DataLoader2(root, path = path, jsonFile = jsonFile, mode = mode)
    loader = DataLoader(datasetTraining, batch_size=batch_size, shuffle = shuffleBool)
    
    return loader


def main(device = DEVICE):
    root = os.getcwd()
    p_json = os.path.join(root, 'json_paths.json')
    params = tr.paramsLearning()
    
    with open('cfg.yml', 'r') as file:
        cfg = yaml.safe_load(file) 
    trainingInfos = cfg['training']
    
    params.wbName = trainingInfos['wbName']
    params.nbEpoch = trainingInfos['nbEpoch']
    params.lr = trainingInfos['lr']
    params.scheduleBool = trainingInfos['scheduler']
    params.size = trainingInfos['scheduleParams']['size']
    params.gamma = trainingInfos['scheduleParams']['gamma']
    params.freqEval = trainingInfos['frequenceEval']
    params.freqSave = trainingInfos['frequenceSave']
    params.batchSize = trainingInfos['batch']
    #params.shuffleBool = True
    params.wdecay = trainingInfos['loss']['lambdaL2Weights']
    params.L1LossReg = trainingInfos['loss']['l1Reg']
    params.lossScaling = trainingInfos['loss']['lossScaling']
    params.probDataAug = trainingInfos['dataAugment']['prob']
    params.stdSpeed = trainingInfos['dataAugment']['stdSpeed']
    params.stdDeltaPos = trainingInfos['dataAugment']['stdDeltaPos']
    params.minEvalLoss = float('inf')
    
    topk = trainingInfos['topk']
    if topk == -1:
        params.topk = None
    else:
        params.topk = topk
    
    params.pathData = trainingInfos['pathData']
    pathJsonBool = trainingInfos['pathJsonBool']
    batchSizeEval =  trainingInfos['batchSizeVal']
    
    #p_data = '/scratch/users/jpierre/test_new'
    p_data = params.pathData
    p_training = os.path.join(p_data, 'training/torch_file')
    p_sim = os.path.join(p_data, 'validation/np_file')
    
    if pathJsonBool:
        loaderTraining = getLoader(p_data, batch_size = params.batchSize, jsonFile = p_json, mode = 'training')

        loaderEval = getLoader(p_data, batch_size = batchSizeEval, jsonFile = p_json, mode = 'validation')
    else:
        loaderTraining = getLoader(p_data, batch_size = params.batchSize, mode = 'training')

        loaderEval = getLoader(p_data, batch_size = batchSizeEvale, mode = 'validation')
    

    loaderSim = dl.simLoader2(p_sim)   # possible issue here (update the saving of files)
    loaderSim = DataLoaderPy(loaderSim, batch_size=1, shuffle=False)
    
    
    params.loader = loaderTraining
    params.evalLoaderTorch = loaderEval
    params.evalLoaderSim = loaderSim
    
    
    schedule = [params.size, params.gamma]
    #if 
    params.schedule = schedule
    
    modelName = trainingInfos['modelName']
    featureShape = cfg['feature']['inShape']
    edgeShape = cfg['feature']['edgeShape']
    model = tr.loadModel(modelName, featureShape, edgeShape)
    
    model = model.to(device)
    params.optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay= params.wdecay)
    
    wandb.init(project = 'master_thesis', name = f"{params.wbName}")
                 
    
    depo_trained = os.path.join(root, 'model_trained')
    if not os.path.exists(depo_trained):
        os.makedirs(depo_trained)
        
    params.pathSaveEval = os.path.join(depo_trained, trainingInfos['evalModel'])
    params.pathSaveFull = os.path.join(depo_trained, trainingInfos['saveModel'])
    
    
    model = tr.train_1step(model, params, device = device)
    
                               
                               
                               
    


if __name__ == '__main__':
    main()