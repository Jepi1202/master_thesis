import argparse
import os
import sys
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from typing import Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchData
from torch.utils.data import Dataset as torchDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader as DataLoaderPy
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GeneralConv
from dataLoading import DataGraph, simLoader, simLoader2
import features as ft
from features import FEATURE_SHAPE, EDGE_SHAPE
from NNSimulator import genSim, optimized_getGraph
from trainingStats import getHeatmap, getHeatmap2
from norm import normalizeCol


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('cfg.yml', 'r') as file:
    cfg = yaml.safe_load(file) 
    
## Normalization
NB_HIST = cfg['feature']['nbHist']
FEATURE_TYPE = cfg['feature']['featureType']

MIN_X = cfg['normalization']['position']['minPos']
MAX_X = cfg['normalization']['position']['maxPos']
MIN_Y = cfg['normalization']['position']['minPos']
MAX_Y = cfg['normalization']['position']['maxPos']

MIN_RAD = cfg['normalization']['radius']['minRad']
MAX_RAD = cfg['normalization']['radius']['maxRad']

R = cfg['simulation']['parameters']['R']
LIMIT_DIST = 0.95 * cfg['simulation']['parameters']['boundary']

PATH_MODEL = '/home/jpierre/v2/models'
PATH_RESULT = '/home/jpierre/v2/results'                                    # to adapt
PATH_DEPO_MODELS = '/home/jpierre/v2/trained_models'                        # to adapt
SYS_PATH = sys.path.copy()

#NB_ROLL = cfg['training']['rolloutNb']
NB_ROLL = 1


SPACE_INDICES = []
for i in range(NB_HIST):
    SPACE_INDICES.append([0+4*i, 1+4*i])
    
#SPACE_INDICES = [[0,1],[4,5]]                                                       # X, Y elements in node features
EDGE_INDICES = None                                                        # X, Y elements in edge features

#TODO displaying parameters of the learning
class paramsLearning():
    """ 
    Parameters of the learning
    """

    def __init__(self,lodaerLearning, loaderSim,loaderEval, nbEpoch:int = 200, wbName: Optional[Union[str, None]] = None,  lr:float = 0.005, wdecay:float = 5e-4, schedule:Optional[Union[tuple, None]] = None, L1LossReg:bool = 0, dataAug:bool = False):

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
        

#TODO adapt to change the cfg
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


def loadModel(modelName:str, inputShape:int = FEATURE_SHAPE, edges_shape = None, path = PATH_MODEL):
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


def setUp(model, wbName, lr:float = 0.005, wdecay:float = 5e-4, schedule:Optional[Union[tuple, None]] = None):
    """ 
    Function to initialize the learning

    Args:
    -----
        - `model`:
        - `wbName`:
        - `lr`:
        - `wdecay`:
        - `schedule`:

    Returns:
    --------
        the optimizer and the scheduler
    """
    wandb.init(project = 'master_thesis', name = f"{wbName}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)

    if schedule is None:
        lr_scheduler = None
    else:
        lr_scheduler = StepLR(optimizer=optimizer, step_size=schedule[0], gamma=schedule[1])  # init: 100, 0.5

    #wandb.watch(model, log = 'all', log_freq=100)
    model.train()

    return optimizer, lr_scheduler


def dataAugFun(data, device = DEVICE, indices = SPACE_INDICES, indices2 = EDGE_INDICES):
    """ 
    Data augmentation function 
    """
    if FEATURE_TYPE == 'full':
        p1 = np.random.random()
        if p1 < 0.5:   #translation
            rdNb = np.random.uniform(-500, 500, (1,2))
            c = torch.from_numpy(np.tile(rdNb, (data.x.shape[0],1))).to(device)
            
            if indices is not None:
                for j in range(len(indices)):
            
                    data.x[:, indices[j]] += c
            
            if indices2 is not None:
                c = torch.from_numpy(np.tile(rdNb, (data.edge_attr.shape[0],1))).to(device)
                data.edge_attr[:, indices2] += c

        # lacks to apply that for the features of the edges ...
        
        p2 = np.random.random()
        if p2 < 0.8:   #translation
            rdNb = torch.normal(mean=0, std=0.001, size=data.x.shape).to(device)
            rdNb2 = torch.normal(mean=0, std=0.001, size=data.edge_attr.shape).to(device)
            
            data.x += rdNb
            data.edge_attr += rdNb2

    return data


def normalizationFun(data, indices = SPACE_INDICES, indices2 = EDGE_INDICES):
    """
    Function to normalize data
    """
    if FEATURE_TYPE == 'full':
        # normalize node features
        
        if indices is not None:
            for j in range(len(SPACE_INDICES)):
                
                data.x[:, SPACE_INDICES[j]] = normalizeCol(data.x[:, SPACE_INDICES[j]], MIN_X, MAX_X)


        # normalize edge features
        if indices2 is not None:
            data.edge_attr[:, indices2] = normalizeCol(data.edge_attr[:, indices2], MIN_X, MAX_X)
            
    return data


#TODO adapt with gamma for puting more weight on first simulations of rollout
def lossFun(preds:torch.tensor, gts:torch.tensor, binaries:Optional[Union[None, torch.tensor]] = None, topk = None):
    """ 
    Function to perform the loss

    Args:
    -----
        - `preds`: predictions
        - `gts`: ground truths
        - `binaries`: binaries for selection
        - `topk`:
    """

    # compute the L2 distance
    res = (preds - gts)**2

    # only take the elements that are definded wrt binaries
    if binaries is not None:
        res = res * binaries
        
    # only select the k worst losses
    res = res.view(-1)
    if topk is not None:
        res, _ = torch.topk(res, topk)
        

    #return F.mse_loss(preds.view(-1), gts.view(-1))
    return torch.mean(res)


def train(model, optimizer, lr_scheduler,  paramLearning, path_result_LAST, path_result_MIN, device = DEVICE):

    # might need to put the wandb init here (depends how it is coded)
    #optimizer, lr_scheduler = setUp(model, 'test_new_run_b')


    ## GET everything from the paramLearning class
    loader = paramLearning.lodaerLearning
    loaderEval = paramLearning.loaderEval
    #loaderSim = paramLearning.Sim
    nbEpoch = paramLearning.nbEpoch
    dataAug = paramLearning.dataAug


    model.train()

    i = 0
    j = 0
    k = 0
    cumuLoss = 0

    minEvalLoss = float('inf')
    for epoch in range(nbEpoch):
            
        for data, idx in tqdm(loader):
            
            optimizer.zero_grad()
            
            bina = (torch.abs(data.x[:, 0]) <= 175) & (torch.abs(data.x[:, 1]) <= 175)
            bina = bina.repeat(2*NB_ROLL, 1).swapaxes(0,1).to(DEVICE)
            
            data = data.to(DEVICE)
            if i == 0:
                print(data.ptr)
                print(data.x.shape)
            y = data.y
            
            if dataAug:
                data = dataAugFun(data)
                
            pos = data.x[:, :2].clone()    # keep the comp graph ?
                    
            #data = normalizationFun(data)
            
            #_, out = genSim(model, NB_ROLL, data, pos, train = True)
            out = model(data)
            j = j+1


            loss0 = 100* lossFun(out, y[:, :(2 * NB_ROLL)], bina, topk = None)
            loss1 = 0.001*model.L1Reg(data)
            loss = loss0 + loss1
            
            cumuLoss = cumuLoss + loss
            
            if ((j+1) % 32) == 0:
                
                
                max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                
                cumuLoss.backward()
                optimizer.step()
                
                max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
                wandb.log({'max_gradVal':max_grad_value})
                
                cumuLoss = 0
                
                       
            
            if ((i+1) % 25 == 0 or i == 0):
                
                max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                wandb.log({'step': i, 'epoch':epoch, 'Training Loss': loss.item(), 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                #loss = 0
                
            #loss.backward()
            #optimizer.step()
            
            #max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
            #wandb.log({'max_gradVal':max_grad_value})
            
            
            
            if ((i+1) % 500 == 0 or i == 0):
                i = i+1
                k += 1
                #lr_scheduler.step()

                #v = eval(model, loaderEval, path_result_MIN)
                ###


                model.eval()
                with torch.no_grad():

                    evalLoss = 0
                    
                    for d, _ in loaderEval:
                        
                        d = torch.squeeze(d, dim = 0).numpy()
                        
                        x, y = ft.getFeatures(d.copy(), np.array([normalizeCol(R, MIN_RAD, MAX_RAD)]), nb = 4)
                        attr, inds = optimized_getGraph(d[5, :, :].copy())
                        s = Data(x = x[4], edge_attr = attr, edge_index = inds).to(DEVICE)
                        res = genSim(model, 80, s, torch.from_numpy(d[5, :, :].copy()).float(), train = False)

                        
                        evalLoss += F.l1_loss(res.reshape(-1), torch.from_numpy(d[5:86, :, :].copy()).reshape(-1).to(DEVICE))
                        

                    #nb grad is none for normalization layer 

                    if evalLoss < minEvalLoss:
                        minEvalLoss = evalLoss
                        torch.save(model.state_dict(), path_result_MIN)
                    
                    wandb.log({'step': i,'eval_loss': evalLoss / len(loaderEval)})
                    
                    pathSim = '/home/jpierre/v2/part_1_b/training/np_file/output_0.npy'

                    getHeatmap2(pathSim, model)
                    
                    wandb.log({'heatmap':wandb.Image(plt)})
                    
                    plt.close()
                    
                    """
                    if k % 5 == 0:
                        
                        x, y = ft.getFeatures(d.copy(), np.array([normalizeCol(R, MIN_RAD, MAX_RAD)]), nb = 4)
                        attr, inds = optimized_getGraph(d[5, :, :].copy())
                        s = Data(x = x[4], edge_attr = attr, edge_index = inds).to(DEVICE)
                        create_simulation_video_cv2(d[5:, :, :],f'/home/jpierre/v2/part_1_b/vids_train/base_{k}.mp4')
                        res = genSim(model, 1000, s, torch.from_numpy(d[5, :, :].copy()).float(), train = False)
                        create_simulation_video_cv2(res.cpu().detach().numpy(), f'/home/jpierre/v2/part_1_b/vids_train/sim_{k}.mp4')
                    """
                        
                    """    
                    if k % 10:
                        x, y = ft.getFeatures(sim.copy(), np.array([normalizeCol(R, MIN_RAD, MAX_RAD)]), nb = 4)
                        attr, inds = [], []
                        for i in range(len(x)):
                            vd, vi = optimized_getGraph(sim[1+i])
                            attr.append(vd)
                            inds.append(vi)
    
                    """
                        

                model.train()

                ###

                #wandb.log({'step': i,'eval_loss': v})
            
            
            
            if (i % 10000) == 0:
                torch.save(model.state_dict(), path_result_LAST)

            i += 1
            
            
            if ((i+1) % 5000) == 0:
            
                lr_scheduler.step()
        
        
def getLoader(path, batch_size = 32, shuffleBool = True, root = None):
    p_train = os.path.join(path, 'torch_file')
    datasetTraining = DataGraph(root, path = path)
    loader = DataLoader(datasetTraining, batch_size=batch_size, shuffle = shuffleBool)
    
    return loader


def create_simulation_video_cv2(data, filename='simulation.mp4', fps=10, size=(600, 600)):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements using cv2.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - filename: Name of the output MP4 file.
    - fps: Frames per second for the output video.
    - size: Size of the output video frame.
    """
    # Convert the data to numpy for easier manipulation
    data_np = data
    
    # Normalize coordinates to fit within the video frame size
    data_np -= data_np.min(axis=(0, 1), keepdims=True)
    data_np /= data_np.max(axis=(0, 1), keepdims=True)
    data_np *= np.array([size[0] - 1, size[1] - 1])
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(filename, fourcc, fps, size)
    
    for i in range(data_np.shape[0]):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for x, y in data_np[i]:
            # Draw the cell as a circle
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        out.write(frame)
    
    out.release()


if __name__ == '__main__':
    root = os.getcwd()
    p = '/scratch/users/jpierre/test_speed'
    path_training = os.path.join(p, 'training')
    path_evaluation = os.path.join(p, 'validation')
    
    p_train = os.path.join(path_training, 'torch_file')
    print(os.path.exists(p_train))
    loader = getLoader(p_train, batch_size = 1)

    #p_eval = os.path.join(path_evaluation, 'torch_file')
    #loaderEval = getLoader(p_eval)
    p_eval = os.path.join(path_evaluation, 'np_file')
    print(os.path.exists(p_eval))
    loaderEval = simLoader2(p_eval)
    loaderEval = DataLoaderPy(loaderEval, batch_size=1, shuffle=False)
    
    lodaerLearning = loader
    loaderSim= None
    loaderEval = loaderEval
    nbEpoch= 500
    wbName = 'roll'
    lr= 0.001
    wdecay = 1e-6
    #schedule = None
    schedule = [1, 0.25]
    L1LossReg= True
    dataAug = True
    
    params = paramsLearning(lodaerLearning, loaderSim, loaderEval, nbEpoch, wbName, lr, wdecay,schedule, L1LossReg, dataAug)
    
    name ='model_1_3'
    #name ='model2'
    #name = 'gat'
    model = loadModel(name, FEATURE_SHAPE, EDGE_SHAPE)
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model, [0,1])
    model = model.to(DEVICE)
    
    result_model_path = os.path.join(root, 'model_trained')
    path_result_MIN = os.path.join(result_model_path, f'{name}_5_speed_min.pt')
    path_result_LAST = os.path.join(result_model_path, f'{name}_5_speed_last.pt')
    
    
    optimizer, lr_scheduler = setUp(model, wbName, lr, wdecay, schedule)
    
    train(model,optimizer, lr_scheduler, params, path_result_LAST, path_result_MIN)
    
    
    