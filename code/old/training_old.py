import torch
import os 
import sys
import yaml
import wandb
import numpy as np
from typing import Optional, Union
from tqdm import tqdm



from torch.optim.lr_scheduler import StepLR
import features as ft
from norm import normalizeCol
from NNSimulator import genSim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dataLoading import DataGraph



PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 

FEATURE_SHAPE = 9
EDGE_SHAPE = 3
R_PARAM = 0.1
MIN_RAD = cfg['normalization']['radius']['minRad']
MAX_RAD = cfg['normalization']['radius']['maxRad']

print('loading training')

PATH_MODEL = '/home/jpierre/v2/models'
PATH_RESULT = '/home/jpierre/v2/results'                                    # to adapt
PATH_DEPO_MODELS = '/home/jpierre/v2/trained_models' 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class paramsLearning():
    """ 
    Parameters of the learning
    """

    def __init__(self):

        self.wbName = None
        self.nbEpoch = None
        self.lr = None

        self.schedule = None
        self.scheduleBool = False
        self.size = None
        self.gamma = None

        self.freqEval = None
        self.freqSave = None

        self.loader = None
        self.batchSize = 32
        self.shuffleBool = True
        #self.loaderEval = None
        #self.lodaerLearning = None

        self.wdecay = None
        self.L1LossReg = 0
        self.lossScaling = 1
        
        # not implemented
        #self.dataAug = None     # data augmentation function
        #self.lossFun = None


        self.probDataAug = 0
        self.stdSpeed = 0
        self.stdDeltaPos = 0

        self.minEvalLoss = float('inf')
        self.pathSaveEval = None

        self.evalLoaderTorch = None
        self.evalLoaderSim = None

        self.opitmizer = None
        self.limitDist = float('inf')

        self.pathSaveFull = None
        

    def _loadParameters(self, params = None):
        if params is None:
            # default parameters
            pass

        else:
            self.wbName = params[0]
            self.nbEpoch = params[1]
            self.lr = params[2]
            self.schedule = params[3]
            self.scheduleBool = params[4]
            self.size = params[5]
            self.gamma = params[6]
            self.freqEval = params[7]
            self.freqSave = params[8]
            self.loader = params[9]
            self.wdecay = params[10]
            self.L1LossReg = params[11]
            self.lossScaling = params[12]
            self.probDataAug = params[13]
            self.stdSpeed = params[14]
            self.stdDeltaPos = params[15]
            #self.loaderEval = params[16]
            #self.loaderLearning = params[17]

    @property
    def parameters(self):
        print(f'Wandb Name >>>> {self.wbName}')
        print(f'Number of Epochs >>>> {self.nbEpoch}')
        print(f'Learning Rate >>>> {self.lr}')
        print(f'Schedule >>>> {self.schedule}')
        print(f'Schedule Boolean >>>> {self.scheduleBool}')
        print(f'Size >>>> {self.size}')
        print(f'Gamma >>>> {self.gamma}')
        print(f'Frequency of Evaluation >>>> {self.freqEval}')
        print(f'Frequency of Saving >>>> {self.freqSave}')
        print(f'Simulation Loader >>>> {self.loaderSim}')
        print(f'Weight Decay >>>> {self.wdecay}')
        print(f'L1 Loss Regularization >>>> {self.L1LossReg}')
        print(f'Loss Scaling Factor >>>> {self.lossScaling}')
        print(f'Probabiltiy of data augmentation >>>> {self.probDataAug}')
        print(f'Standard deviation of speed >>>> {self.stdSpeed}')
        print(f'Standard deviation displacement >>>> {self.stdDeltaPos}')




def loadModel(modelName:str, inputShape:int = FEATURE_SHAPE, edges_shape = EDGE_SHAPE, path = PATH_MODEL):
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



def setUp(model, paramsLearning:paramsLearning):
    """ 
    Function to initialize the learning

    Args:
    -----
        - `model`:
        - `paramsLearning`:

    Returns:
    --------
        the optimizer and the scheduler
    """

    wbName = paramsLearning.wbName
    lr = paramsLearning.lr
    weightDecay = paramsLearning.wdecay
    scheduleBool = paramsLearning.scheduleBool
    scheduleSize = paramsLearning.size
    scheduleGamma = paramsLearning.gamma

    wandb.init(project = 'master_thesis', name = f"{wbName}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)

    if scheduleBool:
        lr_scheduler = StepLR(optimizer=optimizer, step_size=scheduleSize, gamma=scheduleGamma)
    else:
        lr_scheduler = False
    
    #wandb.watch(model, log = 'all', log_freq=100)
    model.train()

    return optimizer, lr_scheduler



def dataAugFun(data, params:paramsLearning, device = DEVICE):
    """ 
    Data augmentation function
    Implements only the noisy steps on the speeds ...
    """

    pCutoff = params.probDataAug
    stdSpeed = params.stdSpeed
    stdDeltaPos = params.stdDeltaPos
        
    p2 = np.random.uniform(0, 1)
    if p2 < pCutoff:   #noise
        rdNb = torch.normal(mean=0, std=stdSpeed, size=data.x.shape).to(device)
        #TODO: update not change R ...
        rdNb2 = torch.normal(mean=0, std=stdDeltaPos, size=data.edge_attr.shape).to(device)

        data.x += rdNb
        data.edge_attr += rdNb2

    return data



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
        
    return torch.mean(res)


# WHAT IS ROOT ....
def getTorchLoader(path, params:paramsLearning, root = None):
    """ 
    Function to create a pytorch geometric loader

    Args:
    ----
        - path
        - params

    Returns:
    --------
        pytorch geometric dataloader
    """

    batch_size = params.batchSize
    shuffleBool = params.shuffleBool
    datasetTraining = DataGraph(root, path = path)
    loader = DataLoader(datasetTraining, batch_size=batch_size, shuffle = shuffleBool)
    
    return loader



def evaluate(model, params:paramsLearning, device = DEVICE):
    
    minEvalLoss = params.minEvalLoss
    path_result_MIN = params.pathSaveEval
    loaderTorch = params.evalLoaderTorch
    loaderSim = params.evalLoaderSim

    evalLoss = 0
    evalLossSim = 0
    model.eval()

    with torch.no_grad():

        if loaderTorch is not None:
            evalLoss = 0

            for d, _ in loaderTorch:
                d = d.to(device)
                res = model(d)  
                
                evalLoss += torch.nn.functional.l1_loss(res.reshape(-1), d.y[:, :2].reshape(-1))
                
                break
                

            #nb grad is none for normalization layer 

            if evalLoss < minEvalLoss:
                minEvalLoss = evalLoss
                torch.save(model.state_dict(), path_result_MIN)
            
            #wandb.log({'step': i,'eval_loss': evalLoss / len(loaderTorch)})



        if loaderSim is not None:


            for d, _ in loaderSim:
                        
                d = torch.squeeze(d, dim = 0).numpy()
                
                # would be great a function where input simulation and output the predicted stuff
                x, y = ft.getFeatures(d.copy(), np.array([normalizeCol(R_PARAM, MIN_RAD, MAX_RAD)]), nb = 4)
                attr, inds = ft.optimized_getGraph(d[5, :, :].copy())
                s = Data(x = x[4], edge_attr = attr, edge_index = inds).to(DEVICE)
                res = genSim(model, 80, s, torch.from_numpy(d[5, :, :].copy()).float(), train = False)
                                
                evalLossSim += torch.nn.functional.l1_loss(res.reshape(-1), torch.from_numpy(d[5:86, :, :].copy()).reshape(-1).to(DEVICE))


        # other representation methods here ...

    model.train()        
    return evalLoss, evalLossSim



def train_1step(model, params:paramsLearning, device = DEVICE, debug:bool = True):

    loader = params.loader
    nbEpoch = params.nbEpoch
    optimizer = params.optimizer
    limitDist = params.limitDist

    topk = params.topk
    scaleLoss = params.lossScaling
    scaleL1 = params.L1LossReg

    pathSave = params.pathSaveFull


    model.train()
    i = 0

    for epoch in range(nbEpoch):
        for data, _ in tqdm(loader):
            optimizer.zero_grad()
            #bina = (torch.abs(data.x[:, 0]) <= limitDist) & (torch.abs(data.x[:, 1]) <= limitDist)
            #bina = bina.repeat(2, 1).swapaxes(0,1).to(device)
            bina = None

            data = data.to(device)
            y = data.y

            data = dataAugFun(data, params = params)

            out = model(data)


            loss0 = scaleLoss * lossFun(out, y[:, :2], bina, topk = topk)
            loss1 = scaleL1 * model.L1Reg(data)
            loss = loss0 + loss1

            loss.backward()
            optimizer.step()


            if debug:
                max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
                wandb.log({'max_gradVal':max_grad_value})


                if ((i+1) % 25 == 0 or i == 0):
                
                    max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                    #wandb.log({'step': i, 'epoch':epoch, 'Training Loss': loss.item(), 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    wandb.log({'step': i, 'epoch':epoch, 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    

            if ((i+1) % 500 == 0 or i == 0):
                model.eval()
                with torch.no_grad():
                    evalLoss, evalLossSim = evaluate(model, params, device = device)
                    wandb.log({'step': i,'eval_loss': evalLoss, 'sim loss': evalLossSim})
                model.train()



            if (i % 10000) == 0:
                torch.save(model.state_dict(), pathSave)



            i += 1


    return model


