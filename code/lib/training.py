import torch
import os 
import sys
import yaml
import numpy as np
from typing import Optional, Union
from tqdm import tqdm



from torch.optim.lr_scheduler import StepLR, ExponentialLR
import features as ft
from norm import normalizeGraph
from NNSimulator import genSim, getSimulationData
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dataLoading import DataGraph


PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, 'cfg.yml'), 'r') as file:
    cfg = yaml.safe_load(file) 

FEATURE_SHAPE = 8
EDGE_SHAPE = 5
R_PARAM = 1
MIN_RAD = cfg['normalization']['radius']['minRad']
MAX_RAD = cfg['normalization']['radius']['maxRad']
BOUNDARY = cfg['simulation']['parameters']['boundary']



print('loading training')

PATH_MODEL = '/home/jpierre/v2/models'
PATH_RESULT = '/home/jpierre/v2/results'                                    # to adapt
PATH_DEPO_MODELS = '/home/jpierre/v2/trained_models' 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from measure import Param_eval, EvaluationCfg, evaluateLoad, saveLoader
import wandb


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import cv2


def plt2frame(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    plt.close(fig)

    return frame


def getSparsityPlot(messages, videoOut = True):

    #stdMessage = np.std(messages,axis = -1)
    stdMessage = np.std(messages,axis = 0)
    
    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(stdMessage[np.argsort(stdMessage)[::-1][None, :15]], cmap='gray_r', edgecolors='k')
    plt.axis('off')
    plt.grid(True)
    ax.set_aspect('equal')
    plt.text(15.5, 0.5, '...', fontsize=30)
    plt.tight_layout()

    if videoOut:
        return plt2frame(fig)
    
    else:
        return None


def create_mp4_with_frames(output_path, frames, fps=10):
    if not frames:
        raise ValueError("No frames to write to video.")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()


def combine_video_with_new_frame(input_path, output_path, new_frame):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    frames.append(new_frame)
    create_mp4_with_frames(output_path, frames)



def getVideo(model, data, device = DEVICE):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        messages = model.GNN.message(None, None, data.edge_attr).cpu().detach().numpy()

        p = os.path.join(os.getcwd(), 'video_l1_bar.mp4')
        frame = getSparsityPlot(messages)

        if not os.path.exists(p):
            create_mp4_with_frames(p, [frame])
        else:
            combine_video_with_new_frame(p, p, frame)


def save_checkpoint(step,
                    model, 
                    optimizer, 
                    scheduler,
                    path):


    checkpoint = {
        'step': step,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }


    torch.save(checkpoint, path)


class PersScheduler():
    def __init__(self, optimizer, type:str, params = None, verbose:bool = True):

        self.type = type
        self.params = params
        self.verbose = verbose

        if type == 'exp':

            gamma = params
            if params is None:
                gamma = 0.9

            if verbose:
                print(f'INFO >>>>> Exponential scheduler with gamma <{gamma}>')

            self.scheduler = ExponentialLR(optimizer, gamma=params)
                

        elif type == 'no':
            if verbose:
                print(f'INFO >>>>> No scheduler')

            self.scheduler = None

        else:
            print('ERROR >>>>  NO KNOWN SCHDULER --- using NONE')
            self.scheduler = None


    def getScheduler(self):
        return self.scheduler
    

    def stepScheduler(self, fun = None, params = None):
        if fun is not None:
            fun(self.scheduler, *params)

        else:
            if self.scheduler:
                self.scheduler.step()


class DataAugBloc():
    def __init__(self, fun = None, params = None):

        self.fun = fun
        self.params = params

    def augmentData(self, data):
        if self.fun is None:
            return data
        
        return self.fun(data, *self.params)
        

class PredictionBloc():
    def __init__(self, params):
        pass


class paramsLearning():
    """ 
    Parameters of the learning
    """

    def __init__(self):

        self.nbEpoch = None
        self.batchSize = 16
        self.shuffleBool = True
        self.nbRoll = None


        ## model elements

        self.modelName = None
        self.model_dict = None
        

        ## evaluate elements

        self.freqEval = None
        self.minEvalLoss = float('inf')


        ## loss elements

        self.type_training = None
        self.number_rollouts_training = None
        self.L1LossReg = 0
        self.lossScaling = 1
        self.topk = None


        ## Data elements

        self.pathData = None
        self.loader = None
        self.evalLoaderTorch = None
        self.evalLoaderSim = None
        self.dt_add = False                     ##########
        self.dt = None                          ##########

        ## Optimizer elements

        self.optimizer_type = None
        self.lr = None
        self.wdecay = None
        self.optimizer = None


        ## scheduler elements

        self.schedule = None


        ## Data augmentation elements

        self.augBool = 1
        self.probDataAug = 0
        self.stdSpeed = 0
        self.stdDeltaPos = 0


        ## Save elements

        self.freqSave = None
        self.wbName = None
        self.pathSaveEval = None
        self.pathSaveFull = None

        
        ## Other (not used)
        


        self.limitDist = float('inf')

        self.gammaLoss = [0.95 ** i for i in range(80)]
        self.batchRollout = None
        
        
        self.displayBoolAug = 1

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
        params = {
            attr: getattr(self, attr) for attr in dir(self) 
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }
        for key, value in params.items():
            print(f'{key}: {value}')



def dataAugFun(data, params, device = 'cpu'):
    """ 
    Data augmentation function
    Implements only the noisy steps on the speeds ...
    """


    pCutoff = params.probDataAug
    augBool = params.augBool
    
    
    if params.displayBoolAug:
        print('>>>>>>> INFOS: data augmentation:')
        print(f'probability of the data augmentation >>> {pCutoff}')
        print(f'activation of the data augmentation >>> {augBool}')
        params.displayBoolAug = False
        
    if augBool:
        stdSpeed = 0.01 * torch.abs(data.x)
        meanSpeedNormal = torch.zeros_like(data.x)
        stdDeltaPos = 0.02
        stdCos = 0.002
        stdRadius = 0.0005

        p2 = np.random.uniform(0, 1)
        if p2 < pCutoff:   #noise
            rdNb = torch.normal(mean=meanSpeedNormal, std=stdSpeed).to(device)
            rdNb2 = torch.normal(mean=0, std=stdDeltaPos, size=(data.edge_attr.shape[0], 1), device=device)
            rdNb3 = torch.normal(mean=0, std=stdCos, size=(data.edge_attr.shape[0], 2), device=device)
            rdNb4 = torch.normal(mean=0, std=stdRadius, size=(data.edge_attr.shape[0], 2), device=device)

            data.x += rdNb
            data.edge_attr[:, 0] += rdNb2.squeeze()
            data.edge_attr[:, 1:3] += rdNb3
            data.edge_attr[:, 3:] += rdNb4


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

    # compute the L1 distance
    res = torch.square(preds - gts)
    #res = torch.square(preds - gts) / (gts ** 2)

    # only take the elements that are definded wrt binaries
    if binaries is not None:
        res = res * binaries
        
    # only select the k worst losses
    res = res.view(-1)
    if topk is not None:
        topkC = int(res.shape[0] * topk)
        res, _ = torch.topk(res, topkC)
        
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

    
    cfg_eval = EvaluationCfg()
    cfg_eval.norm_angleError = Param_eval(wandbName = 'fdsf')
    #cfg_eval.L1_vect = Param_eval(wandbName = 'Std message')
    cfg_eval.degree_error = Param_eval(wandbName = 'Degree error')
    cfg_eval.dist_error = Param_eval(wandbName = 'Distance error')
    model.eval()
    
    evalLossSim = 0

    with torch.no_grad():

        if loaderTorch is not None:
            res = evaluateLoad(loaderTorch, model, cfg_eval, device = device) 
            
            evalLoss = res['evalLoss']

            if evalLoss < minEvalLoss:
                minEvalLoss = evalLoss
                torch.save(model.state_dict(), path_result_MIN)
            
            #wandb.log({'step': i,'eval_loss': evalLoss / len(loaderTorch)})
            
            saveLoader(res, cfg_eval)


            for data_ev, _ in loaderTorch:
                getVideo(model, data_ev)
                break



        if loaderSim is not None:


            for d, _ in loaderSim:
                        
                d = torch.squeeze(d, dim = 0).numpy()
                start = 8       # not 0
                res = getSimulationData(model, 80, d, i = start)
                L = res.shape[0]
                                
                evalLossSim += torch.nn.functional.l1_loss(res.reshape(-1), torch.from_numpy(d[start:(start + L), :, :].copy()).reshape(-1).to(DEVICE))


        # other representation methods here ...

    model.train()        
    return evalLoss, evalLossSim



def train_1step(model, params:paramsLearning, device = DEVICE, debug:bool = True):

    loader = params.loader
    nbEpoch = params.nbEpoch
    optimizer = params.optimizer
    limitDist = params.limitDist
    schedule = params.schedule
    
    topk = params.topk
    print(f'topk >>>> {topk}')
    scaleLoss = params.lossScaling
    scaleL1 = params.L1LossReg

    pathSave = params.pathSaveFull

    cfg_save = params.cfg_save


    model.train()
    i = 0

    for epoch in range(nbEpoch):
        for data, _ in tqdm(loader):
            optimizer.zero_grad()
            #bina = (torch.abs(data.x[:, 0]) <= limitDist) & (torch.abs(data.x[:, 1]) <= limitDist)
            #bina = bina.repeat(2, 1).swapaxes(0,1).to(device)

            data = data.to(device)
            pos = data.x[:, :2]

            # limitDist ===  boundary now ...
            #bina = (torch.abs(pos[:, 0]) <= 120) & (torch.abs(pos[:, 1]) <= 120)
            #bina = bina.repeat(2, 1).swapaxes(0,1).to(device)

            data.x = data.x[:, 2:]

            bina = None
            
            y = data.y

            data = dataAugFun(data, params = params)
            data = normalizeGraph(data)

            out = model(data)
            
            if params.dt_add:
                nextPose = pos + out * params.dt
            else:
                nextPose = pos + out
            
            out[:, 0] = torch.where((nextPose[:, 0] < -BOUNDARY) | (nextPose[:, 0] > BOUNDARY), -out[:, 0], out[:, 0])
            out[:, 1] = torch.where((nextPose[:, 1] < -BOUNDARY) | (nextPose[:, 1] > BOUNDARY), -out[:, 1], out[:, 1])
            
            y = torch.swapaxes(y, 0, 1) 


            loss0 = scaleLoss * lossFun(out.reshape(-1), y[0, :, :].reshape(-1), bina, topk = topk)
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
                    

            if ((i+1) % 1500 == 0 or i == 0):
                model.eval()
                with torch.no_grad():
                    evalLoss, evalLossSim = evaluate(model, params, device = device)
                    wandb.log({'step': i,'eval_loss': evalLoss, 'sim loss': evalLossSim})
                model.train()



            if (i % 10000) == 0:
                torch.save(model.state_dict(), pathSave)

                name_checkt_save = f'{cfg_save.split('.')[0]}_step-{i}.pth'
                save_checkpoint(i, model, optimizer, schedule.scheduler, name_checkt_save)



            i += 1
        #schedule.step()
        schedule.stepScheduler()
        print(optimizer.param_groups[0]['lr'])


    return model


def train_roll(model, params:paramsLearning, device = DEVICE, debug:bool = True):
    loader = params.loader
    nbEpoch = params.nbEpoch
    optimizer = params.optimizer
    limitDist = params.limitDist

    topk = params.topk
    scaleLoss = params.lossScaling
    scaleL1 = params.L1LossReg

    pathSave = params.pathSaveFull

    nbRoll = params.nbRoll
    gamma = torch.tensor(params.gammaLoss).to(device)
    batchRollout = params.batchRollout
    # apply some gamma later ...

    cfg_save = params.cfg_save



    model.train()
    i = 0
    cumLoss = 0
    
    #print('test0')

    for epoch in range(nbEpoch):
            
        for data, idx in tqdm(loader):
            
            #print('test1')
            
            optimizer.zero_grad()

            nb_rolling = np.random.randint(1, nbRoll+1)
            
            data = data.to(DEVICE)            

            pos  = data.x[:, :2]
            data.x = data.x[:, 2:]
            y = data.y
            
            #print(y.shape)


            bina = None
            
            #print('test2')
            #data = dataAugFun(data, params = params)
            data = normalizeGraph(data)


            #### modify for the dt update
            if nb_rolling > 1:                 
                _, out = genSim(model, nb_rolling, data, pos, train = True)
                
                #print(out.shape)
            else:
                out = model(data)
                
            #print('test3')
            y = torch.swapaxes(y, 0, 1)   
            

            loss0 = scaleLoss * lossFun(out.reshape(-1), y[:nb_rolling, :, :].reshape(-1), bina, topk = topk)
            loss1 = scaleL1 * model.L1Reg(data)
            loss = loss0 + loss1

            cumLoss += loss


            if (i+1)%batchRollout == 0:
                cumLoss.backward()
                optimizer.step()

                if debug:
                    max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
                    wandb.log({'max_gradVal':max_grad_value})

                cumLoss = 0



            if debug:

                if ((i+1) % 25 == 0 or i == 0):
                
                    max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                    wandb.log({'step': i, 'epoch':epoch, 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    


            if ((i+1) % 10000 == 0 or i == 0):
                model.eval()
                with torch.no_grad():
                    evalLoss, evalLossSim = evaluate(model, params, device = device)
                    wandb.log({'step': i,'eval_loss': evalLoss, 'sim loss': evalLossSim})
                model.train()


            if (i % 10000) == 0:
                torch.save(model.state_dict(), pathSave)

                name_checkt_save = f'{cfg_save.split('.')[0]}_step-{i}.pth'
                save_checkpoint(i, model, optimizer, schedule.scheduler, name_checkt_save)

            
            i += 1
            

    return model