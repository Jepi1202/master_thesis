import argparse
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import wandb
import torch.utils.data as torchData
from torch.utils.data import Dataset as torchDataset
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GeneralConv
from typing import Optional, Union
from torch.optim.lr_scheduler import StepLR

from features import FEATURE_SHAPE

# personnal imports
import features as ft       # feature library


PATH_MODEL = '/home/jpierre/v2/models'
PATH_RESULT = '/home/jpierre/v2/results'                                    # to adapt
PATH_DEPO_MODELS = '/home/jpierre/v2/trained_models'                        # to adapt
SPACE_INDICES = [0,1]                                                       # X, Y elements in node features
EDGE_INDICES = [...]                                                        # X, Y elements in edge features

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SYS_PATH = sys.path.copy()


MIN_X = -150
MAX_X = 150
MIN_Y = -150
MAX_Y = 150


def normalizeCol(vect: np.array, minVal:float, maxVal:float)->np.array:
    """ 
    Function used in order to apply min-max stand on features

    Args:
    -----
        -`vect` (np.array): array to normalize
        - `minVal` (float): min value in min-max stand
        - `maxVal` (float): max value in min-max stand

    Returns:
    --------
        the normalized vector
    """
    assert minVal < maxVal
    
    ran = maxVal - minVal
    return (vect - minVal)/ran


def createFolder(path:str)->None:
    """
    Function to make a directory
    
    Args:
    -----
        - `path`: path of the new directory
    """
    
    assert os.path.exists(path) == False
    
    os.makedirs(path)

def getTime():
    infos = datetime.now()

    day = infos.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
    time = infos.strftime("%H:%M:%S")  # Format as HH:MM:SS

    timeInfos = f"{day}__{time}"

    return timeInfos


class paramsLearning():
    """ 
    Parameters of the learning
    """
    def __init__(self,lodaerLearning, loaderSim,loaderEval, nbEpoch:int = 200, wbName: Optional[Union[str, None]] = None,  lr:float = 0.005, wdecay:float = 5e-4, schedule:Optional[Union[tuple, None]] = None, L1LossReg:bool = 0, dataAug:bool = False):

        if wbName is None:
            timeInfos = getTime()
            wbName = f'run_{timeInfos}'

        self.wbName = wbName


        self.nbEpoch = nbEpoch

        self.lr = lr


        self.wdecay = wdecay
        
        if schedule is None:
            self.scheduleBool = False
            self.size = 0
            self.gamma = 0.5

        else:
            self.scheduleBool = True
            self.size = schedule[0]
            self.gamma = schedule[1]


        self.freqEval = 500


        self.freqSave = 10000

        
        self.L1LossReg = L1LossReg


        self.loaderSim = loaderSim
        self.loaderEval = loaderEval

        self.lodaerLearning = lodaerLearning


        self.dataAug = dataAug
        


def retrieveArgs():
    """ 
    Function to retrieve the args sent to the python code
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, default= None,
                        help="Model name used for saving the model after the training")
    parser.add_argument("--iter", type=int, default = 200,
                        help="Number of iterations to perform in the learning")
    parser.add_argument("--l1", type=int, default= 0,
                        help="Bool to force the l1 regularisation")
    parser.add_argument("--loss", type=str, default= 'mse',
                        help="Bool to force the l1 regularisation")
    parser.add_argument("--wb", type=str, default= 'new-run',
                        help="Name of the wandb run")
    

    args = vars(parser.parse_args())

    return args



def loadModel(modelName:str, inputShape:int = FEATURE_SHAPE, path = PATH_MODEL):
    """ 
    Old import to get the mdoel
    """

    sys.path.append(path)

    loadFun = __import__(f'{modelName}', fromlist = ('loadNetwork'))


    model = loadFun.loadNetwork(inputShape)

    return model




def setUp(model, wbName, lr:float = 0.005, wdecay:float = 5e-4, schedule:Optional[Union[tuple, None]] = None):
    wandb.init(project = 'master_thesis', name = f"{wbName}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    if schedule is None:
        lr_scheduler = None
    else:
        lr_scheduler = StepLR(optimizer=optimizer, step_size=schedule[0], gamma=schedule[1])  # init: 100, 0.5

    wandb.watch(model, log = 'all', log_freq=100)
    model.train()

    return optimizer, lr_scheduler



def eval(model, loaderEval, path_result_MIN):
    model.eval()
    with torch.no_grad():

        evalLoss = 0

        for d, _ in loaderEval:
            d = d.to(DEVICE)
            out = model(d)

            evalLoss += F.l1_loss(out.reshape(-1), d.y.reshape(-1))


        #nb grad is none for normalization layer 

        if evalLoss < minEvalLoss:
            minEvalLoss = evalLoss
            torch.save(model.state_dict(), path_result_MIN)


    model.train()

    return evalLoss / len(loaderEval)


def dataAugFun(data, device = DEVICE, indices = SPACE_INDICES):
    p1 = np.random.random()
    if p1 < 0.0:   #translation
        c = torch.from_numpy(np.tile(np.random.uniform(-600, 600, (1,2)), (data.x.shape[0],4))).to(device)
        data.x[:, :8] = (data.x[:, :8] + c)

    # lacks to apply that for the features of the edges ...

    return data


def normalizationFun(data, indices = SPACE_INDICES, indices2 = EDGE_INDICES):
    # normalize node features
    data.x[:, indices] = normalizeCol(data.x[:, indices], MIN_X, MAX_X)


    # normalize edge features

    data.edge_attr[:, indices2] = normalizeCol(data.x[:, indices2], MIN_X, MAX_X)


def loss(preds:torch.tensor, gts:torch.tensor, binaries:Optional[Union[None, torch.tensor]] = None):
    """ 
    
    """

    if binaries is None:
        res = (preds - gts)**2
        res = res * binaries
        return torch.mean(res)

    return F.mse_loss(preds.view(-1), gts.view(-1))

def train(model, paramLearning, lossFun, path_result_LAST, path_result_MIN, device = DEVICE):

    # might need to put the wandb init here (depends how it is coded)
    optimizer, lr_scheduler = setUp(model, 'test_new_run_b')


    ## GET everything from the paramLearning class
    loader = paramLearning.lodaerLearning
    loaderEval = paramLearning.loaderEval
    loaderSim = paramLearning.Sim
    nbEpoch = paramLearning.nbEpoch
    dataAug = paramLearning.dataAug


    model.train()

    i = 0
    j = 0

    minEvalLoss = float('inf')
    for epoch in range(nbEpoch):
            
        for data, idx in tqdm(loader):
            
            optimizer.zero_grad()

            # get the binaries
            bina = (torch.abs(data.x[:, 0]) <= 35) & (torch.abs(data.x[:, 1]) <= 35)
            bina = bina.repeat(2, 1).swapaxes(0,1).to(device)
            
            data = data.to(DEVICE)
            
            # data augmentation
            
            if dataAug:
                data = dataAugFun(data)

            # data normalization
            data = normalizationFun(data)
            
            
            out = model(data)        
            loss = lossFun(out, data.y) + model.L1Reg(data)
                       
            
            if ((i+1) % 25 == 0 or i == 0):
                
                max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                wandb.log({'step': i, 'epoch':epoch, 'Training Loss': loss.item(), 'Max_paramVal':max_param_value})
                #loss = 0
                
            loss.backward()
            optimizer.step()
            
            max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
            wandb.log({'max_gradVal':max_grad_value})
            
            
            
            if ((i+1) % 500 == 0 or i == 0):
            

                #v = eval(model, loaderEval, path_result_MIN)
                ###


                model.eval()
                with torch.no_grad():

                    evalLoss = 0

                    for d, _ in loaderEval:
                        
                        bina = (torch.abs(d.x[:, 0]) <= 35) & (torch.abs(d.x[:, 1]) <= 35)
                        bina = bina.repeat(2, 1).swapaxes(0,1).to(device)
                        
                        d = normalizationFun(d)
                        d = d.to(device)
                        out = model(d)

                        evalLoss += F.l1_loss((bina*out).reshape(-1), (bina * d.y).reshape(-1))


                    #nb grad is none for normalization layer 

                    if evalLoss < minEvalLoss:
                        minEvalLoss = evalLoss
                        torch.save(model.state_dict(), path_result_MIN)

                    wandb.log({'step': i,'eval_loss': evalLoss / len(loaderEval)})

                model.train()

                ###

                #wandb.log({'step': i,'eval_loss': v})
            

            
            if (i % 10000) == 0:
                torch.save(model.state_dict(), path_result_LAST)

            i += 1
            j += 1
        
        lr_scheduler.step()



