import torch
import os 
import sys
import yaml
import numpy as np
from typing import Optional, Union
from tqdm import tqdm
import pickle


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


def get_name_file(file, name):
    a = file.split('/')
    p = '/' + '/'.join([element for element in a[:-1] if element])

    path_new = os.path.join(p, name)

    return path_new

def mergeData(file, data):
    if not os.path.exists(file):
        return data
    
    else:
        res = np.load(file)
        res = np.concatenate((res, data))

        return res
    
    
def save_add(path_new, data, out_mode = 'numpy'):
   
    if out_mode == 'numpy':
        np.save(path_new, data)
        
        


def save_pickle(path, object):
    with open(path, 'wb') as f:
        pickle.dump(object, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        loaded_arrays = pickle.load(f)
    return loaded_arrays

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



def getVideo(model, data, device = DEVICE, output='frame'):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        messages = model.GNN.message(None, None, data.edge_attr).cpu().detach().numpy()

        if output == 'frame':

            p = os.path.join(os.getcwd(), 'video_l1_bar.mp4')
            frame = getSparsityPlot(messages)

            if not os.path.exists(p):
                create_mp4_with_frames(p, [frame])
            else:
                combine_video_with_new_frame(p, p, frame)

        else:
            return messages



def save_checkpoint(step,
                    model, 
                    optimizer, 
                    scheduler,
                    path,
                    debug = True):

    if debug:
        return None

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

        self.wbName = None
        self.nbEpoch = None
        self.lr = None

        # simplify
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
        self.augBool = 1

        self.minEvalLoss = float('inf')
        self.pathSaveEval = None

        self.evalLoaderTorch = None
        self.evalLoaderSim = None

        self.opitmizer = None
        self.limitDist = float('inf')

        self.pathSaveFull = None

        self.nbRoll = 6      # new
        self.gammaLoss = [0.95 ** i for i in range(80)]
        self.batchRollout = None
        
        

        self.dt_update = 1
        
        
        self.displayBoolAug = 1
        
        
        
        self.saveFile = os.path.join(os.getcwd(), 'save_file')
        if not os.path.exists(self.saveFile):
            os.makedirs(self.saveFile)

        self.first_save = True

        self.trainingList = []
        self.training_file = os.path.join(self.saveFile, 'training_loss.pkl')
        self.angleList = []
        self.angle_file = os.path.join(self.saveFile, 'angle_eval.pkl')
        self.normList = []
        self.norm_file = os.path.join(self.saveFile, 'norm_eval.pkl')
        self.evalList = []
        self.eval_file = os.path.join(self.saveFile, 'eval_eval.pkl')
        self.messageList = []
        self.message_file = os.path.join(self.saveFile, 'message_eval.pkl')

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


    def save(self, i):

        if i != 0 and i != 1:
            
            if self.first_save:
                save_pickle(self.training_file, self.trainingList)
                save_pickle(self.angle_file, self.angleList)
                save_pickle(self.norm_file, self.normList)
                save_pickle(self.eval_file, self.evalList)
                save_pickle(self.message_file, self.messageList)


                self.first_save = False

            else:
                tr_list = load_pickle(self.training_file)
                tr_list.append(self.trainingList)
                save_pickle(self.training_file, tr_list)

                ang_list = load_pickle(self.angle_file)
                ang_list.append(self.angleList)
                save_pickle(self.angle_file, ang_list)


                norm_list = load_pickle(self.norm_file)
                norm_list.append(self.normList)
                save_pickle(self.norm_file, norm_list)


                eval_list = load_pickle(self.eval_file)
                eval_list.append(self.evalList)
                save_pickle(self.eval_file, eval_list)


                message_list = load_pickle(self.message_file)
                message_list.append(self.messageList)
                save_pickle(self.message_file, message_list)


            self.trainingList = []
            self.angleList = []
            self.normList = []
            self.evalList = []
            self.messageList = []


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

"""
def setUp(model, paramsLearning:paramsLearning): 
    Function to initialize the learning

    Args:
    -----
        - `model`:
        - `paramsLearning`:

    Returns:
    --------
        the optimizer and the scheduler


    wbName = paramsLearning.wbName
    lr = paramsLearning.lr
    weightDecay = paramsLearning.wdecay
    scheduleBool = paramsLearning.scheduleBool
    scheduleSize = paramsLearning.size
    scheduleGamma = paramsLearning.gamma

    #wandb.init(project = 'master_thesis', name = f"{wbName}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)

    if scheduleBool:
        #lr_scheduler = StepLR(optimizer=optimizer, step_size=scheduleSize, gamma=scheduleGamma)
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.7)
        print('nfrjkgnvks')
    else:
        lr_scheduler = False
    
    #wandb.watch(model, log = 'all', log_freq=100)
    model.train()

    return optimizer, lr_scheduler
"""

"""
def dataAugFun(data, params:paramsLearning, device = DEVICE):
     
    Data augmentation function
    Implements only the noisy steps on the speeds ...
    

    pCutoff = params.probDataAug
    #stdSpeed = params.stdSpeed
    #stdDeltaPos = params.stdDeltaPos
    #stdSpeed = 0.002
    #stdDeltaPos = 0.05
    #stdCos = 0.01
    
    stdSpeed = 0.0
    stdDeltaPos = 0.0
    stdCos = 0.0
        
    p2 = np.random.uniform(0, 1)
    if p2 < pCutoff:   #noise
        rdNb = torch.normal(mean=0, std=stdSpeed, size=data.x.shape).to(device)
        rdNb2 = torch.normal(mean=0, std=stdDeltaPos, size=(data.edge_attr.shape[0], 1), device=device)
        rdNb3 = torch.normal(mean=0, std=stdDeltaPos, size=(data.edge_attr.shape[0], 2), device=device)

        data.x += rdNb
        data.edge_attr[:, 0] += rdNb2.squeeze()
        data.edge_attr[:, 1:3] += rdNb3


    return data
"""


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




def lossFunRoll(preds:torch.tensor, gts:torch.tensor, binaries:Optional[Union[None, torch.tensor]] = None, topk = None, weights = None):
    """ 
    Function to perform the loss

    Args:
    -----
        - `preds`: predictions
        - `gts`: ground truths
        - `binaries`: binaries for selection
        - `topk`:
    """

    preds = preds.view(-1)
    gt = gt.view(-1)

    # compute the L1 distance
    res = torch.square(preds - gts)
    #res = torch.square(preds - gts) / (gts ** 2)

    if weights is not None:
        res = torch.mean(res, dim = -1)
        res = torch.mean(res, dim = -1)
        res = res * weights

    res = res.view(-1)
        
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



def evaluate_old(model, params:paramsLearning, device = DEVICE):
    
    minEvalLoss = params.minEvalLoss
    path_result_MIN = params.pathSaveEval
    loaderTorch = params.evalLoaderTorch
    loaderSim = params.evalLoaderSim
    
    dt_update = params.dt_update

    evalLoss = 0
    evalLossSim = 0
    model.eval()

    with torch.no_grad():

        if loaderTorch is not None:
            evalLoss = 0

            for d, _ in loaderTorch:
                d = d.to(device)
                
                pos = d.x[:, :2]
                
                d.x = d.x[:, 2:]
                d = normalizeGraph(d)
                res = model(d)   
                y = d.y
                
                nextPose = pos + dt_update * res
            
                res[:, 0] = torch.where((nextPose[:, 0] < -BOUNDARY) | (nextPose[:, 0] > BOUNDARY), -res[:, 0], res[:, 0])
                res[:, 1] = torch.where((nextPose[:, 1] < -BOUNDARY) | (nextPose[:, 1] > BOUNDARY), -res[:, 1], res[:, 1])

                y = torch.swapaxes(y, 0, 1)
                

                evalLoss += torch.nn.functional.l1_loss(res.reshape(-1), d.y[0, :, :].reshape(-1))
                
                #break
                

            #nb grad is none for normalization layer 

            if evalLoss < minEvalLoss:
                minEvalLoss = evalLoss
                torch.save(model.state_dict(), path_result_MIN)
            
            #wandb.log({'step': i,'eval_loss': evalLoss / len(loaderTorch)})



        if loaderSim is not None:
        
        
            for d, _ in loaderSim:
                        
                d = torch.squeeze(d, dim = 0).numpy()
                start = 8       # not 0
                res = getSimulationData(model, 15, d, i = start)
                L = res.shape[0]
                                
                evalLossSim += torch.nn.functional.l1_loss(res.reshape(-1), torch.from_numpy(d[start:(start + L), :, :].copy()).reshape(-1).to(DEVICE))


        # other representation methods here ...

    model.train()        
    return evalLoss, evalLossSim



def evaluate(model, params:paramsLearning, device = DEVICE):
    
    minEvalLoss = params.minEvalLoss
    path_result_MIN = params.pathSaveEval
    loaderTorch = params.evalLoaderTorch
    loaderSim = params.evalLoaderSim

    
    cfg_eval = EvaluationCfg()
    cfg_eval.norm_angleError = Param_eval(wandbName = 'fdsf')
    #cfg_eval.L1_vect = Param_eval(wandbName = 'Std message')
    cfg_eval.degree_error = Param_eval(wandbName = 'Degree error')
    #cfg_eval.dist_error = Param_eval(wandbName = 'Distance error')
    model.eval()
    
    evalLossSim = 0

    with torch.no_grad():

        if loaderTorch is not None:
            res = evaluateLoad(loaderTorch, model, cfg_eval, device = device) 
            
            params.angleList.append(np.median(res['angleError']))
            params.evalList.append(res['evalLoss'].cpu().detach().numpy())
            params.normList.append(np.median(res['normError']))

            #print(len(params.angleList))
            #print(len(params.evalList))
            #print(len(params.normList))
            #print('-------')
            #print(type(np.median(res['angleError'])))
            #print(type(res['evalLoss'].cpu().detach().numpy()))
            #print(type(np.median(res['normError'])))
            
            evalLoss = res['evalLoss']

            if evalLoss < minEvalLoss:
                minEvalLoss = evalLoss
                torch.save(model.state_dict(), path_result_MIN)
            
            #wandb.log({'step': i,'eval_loss': evalLoss / len(loaderTorch)})
            
            saveLoader(res, cfg_eval)


            #for data_ev, _ in loaderTorch:
                #messages = getVideo(model, data_ev, output = 'vals', device = device)

                #params.messageList.append(messages)
                #break



        if loaderSim is not None:
        
        
            for d, _ in loaderSim:
                        
                d = torch.squeeze(d, dim = 0).numpy()
                start = 8       # not 0
                res = getSimulationData(model, 15, d, i = start, device = device)
                L = res.shape[0]
                                
                evalLossSim += torch.nn.functional.l1_loss(res.reshape(-1), torch.from_numpy(d[start:(start + L), :, :].copy()).reshape(-1).to(device))


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

    dt_update = params.dt_update
    print(f'-------{dt_update}---------')

    pathSave = params.pathSaveFull
    
    cfg_save = os.path.join(os.getcwd(), 'model_trained')
    print(f'cfg_save >>>>>>>>>>>>>>>>>>>>>>>>> {cfg_save}')
    
    
    path_result_MIN = params.pathSaveEval


    list_loss = []
    list_eval = []
    list_evalSim = []

    model.train()
    i = 0


    path_save_loss = get_name_file(path_result_MIN, 'loss_list.npy')
    path_save_eval_loss = get_name_file(path_result_MIN, 'eval_list.npy')
    path_save_eval_sim_loss = get_name_file(path_result_MIN, 'eval_sim_list.npy')

    if os.path.exists(path_save_loss):
        os.remove(path_save_loss)

    if os.path.exists(path_save_eval_loss):
        os.remove(path_save_eval_loss)

    if os.path.exists(path_save_eval_sim_loss):
        os.remove(path_save_eval_sim_loss)

    for epoch in range(40):
        print(epoch)
        for data, _ in tqdm(loader):
            optimizer.zero_grad()
            #bina = (torch.abs(data.x[:, 0]) <= limitDist) & (torch.abs(data.x[:, 1]) <= limitDist)
            #bina = bina.repeat(2, 1).swapaxes(0,1).to(device)

            data = data.to(device)
            data = normalizeGraph(data)
            pos = data.x[:, :2]

            # limitDist ===  boundary now ...
            #bina = (torch.abs(pos[:, 0]) <= 120) & (torch.abs(pos[:, 1]) <= 120)
            #bina = bina.repeat(2, 1).swapaxes(0,1).to(device)

            data.x = data.x[:, 2:]

            bina = None
            
            y = data.y

            data = dataAugFun(data, params = params, device=device)

            out = model(data)
            
            nextPose = pos + dt_update * out
            
            out[:, 0] = torch.where((nextPose[:, 0] < -BOUNDARY) | (nextPose[:, 0] > BOUNDARY), -out[:, 0], out[:, 0])
            out[:, 1] = torch.where((nextPose[:, 1] < -BOUNDARY) | (nextPose[:, 1] > BOUNDARY), -out[:, 1], out[:, 1])
            
            y = torch.swapaxes(y, 0, 1) 


            loss0 = scaleLoss * lossFun(out.reshape(-1), y[0, :, :].reshape(-1), bina, topk = topk)
            loss1 = scaleL1 * model.L1Reg(data)
            loss = loss0 + loss1
            
            list_loss.append(loss.item())
        

            params.trainingList.append(loss.cpu().detach().numpy())

            loss.backward()
            optimizer.step()


            if debug:
                max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
                #wandb.log({'max_gradVal':max_grad_value})


                if (((i+1) % 100) == 0 or i == 0):
                
                    max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                    #wandb.log({'step': i, 'epoch':epoch, 'Training Loss': loss.item(), 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    #wandb.log({'step': i, 'epoch':epoch, 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    

            if ((i+1) % 6000 == 0 or i == 0):
                model.eval()
                with torch.no_grad():
                    evalLoss, evalLossSim = evaluate(model, params, device = device)
                    #wandb.log({'step': i,'eval_loss': evalLoss, 'sim loss': evalLossSim})
                model.train()
                
                print(f'EvalLoss -> {evalLoss}')
                print(f'EvalLoss -> {evalLossSim}')
                
                
                
                    
                list_eval.append(evalLoss.item())
                list_evalSim.append(evalLossSim.item())
                
                path_save_loss = get_name_file(path_result_MIN, 'loss_list.npy')
                path_save_eval_loss = get_name_file(path_result_MIN, 'eval_list.npy')
                path_save_eval_sim_loss = get_name_file(path_result_MIN, 'eval_sim_list.npy')
                
                data_list_loss = mergeData(path_save_loss, np.array(list_loss))
                data_list_eval_loss = mergeData(path_save_eval_loss, np.array(list_eval))
                data_list_eval_sim_loss = mergeData(path_save_eval_sim_loss, np.array(list_evalSim))
                
                save_add(path_save_loss, data_list_loss, out_mode = 'numpy')
                save_add(path_save_eval_loss, data_list_eval_loss, out_mode = 'numpy')
                save_add(path_save_eval_sim_loss, data_list_eval_sim_loss, out_mode = 'numpy')
                
                list_loss = []
                list_eval = []
                list_evalSim = []
                

                                     
                                     
            if ((i+1) % 3000 == 0 or i == 0):
                p_s2 = os.path.join(os.getcwd(), 'save.pth')
                save_checkpoint(i, model, optimizer, schedule.scheduler, p_s2, debug = False)
            



            if (i % 10000) == 0:
                torch.save(model.state_dict(), pathSave)
                
                name_checkt_save = f'{cfg_save}_step-{i}.pth'
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
    schedule = params.schedule
    

    topk = params.topk
    scaleLoss = params.lossScaling
    scaleL1 = params.L1LossReg

    pathSave = params.pathSaveFull

    nbRoll_max = params.nbRoll
    n_roll = 1
    gamma = torch.tensor(params.gammaLoss).to(device)
    #batchRollout = params.batchRollout
    batchRollout = 32

    path_result_MIN = params.pathSaveEval

    dt_update = params.dt_update
    print(f'-------{dt_update}---------')
    
    
    cfg_save = os.path.join(os.getcwd(), 'model_trained')
    print(f'cfg_save >>>>>>>>>>>>>>>>>>>>>>>>> {cfg_save}')

    weights = torch.tensor([0.9 ** i for i in range(10)], device = device)

    
    list_loss = []
    list_eval = []
    list_evalSim = []
    
    path_save_loss = get_name_file(path_result_MIN, 'loss_list.npy')
    path_save_eval_loss = get_name_file(path_result_MIN, 'eval_list.npy')
    path_save_eval_sim_loss = get_name_file(path_result_MIN, 'eval_sim_list.npy')

    if os.path.exists(path_save_loss):
        os.remove(path_save_loss)

    if os.path.exists(path_save_eval_loss):
        os.remove(path_save_eval_loss)

    if os.path.exists(path_save_eval_sim_loss):
        os.remove(path_save_eval_sim_loss)


    model.train()
    i = 0
    cumLoss = 0
    
    #print('test0')

    for epoch in range(40):
        print(epoch)

        
        if n_roll < nbRoll_max:
            n_roll = n_roll + 1
                
        print(f'{epoch} - {n_roll}')

        for data, idx in tqdm(loader):
                        
            optimizer.zero_grad()
            
            data = data.to(device)
            data = normalizeGraph(data)

            pos  = data.x[:, :2]
            data.x = data.x[:, 2:]
            y = data.y
            

            bina = None
            
            #data = dataAugFun(data, params = params)

            nb_rolling = np.random.randint(1, n_roll+1)

            if nb_rolling > 1:                 
                _, out = genSim(model, nb_rolling, data, pos, train = True, dt_scale = dt_update, device = device)         # udpated dt
                
            else:
                out = model(data)

                nextPose = pos + dt_update * out
                out[:, 0] = torch.where((nextPose[:, 0] < -BOUNDARY) | (nextPose[:, 0] > BOUNDARY), -out[:, 0], out[:, 0])
                out[:, 1] = torch.where((nextPose[:, 1] < -BOUNDARY) | (nextPose[:, 1] > BOUNDARY), -out[:, 1], out[:, 1])


                
            y = torch.swapaxes(y, 0, 1)   
            
            
            loss0 = scaleLoss * lossFun(out.reshape(-1), y[:nb_rolling, :, :].reshape(-1), bina, topk = topk)
            #loss0 = scaleLoss * lossFunRoll(out.reshape(-1), y[:nb_rolling, :, :].reshape(-1), bina, topk = topk, weigths = weights[:nb_rolling])
            loss1 = scaleL1 * model.L1Reg(data)
            loss = loss0 + loss1
                                     
            list_loss.append(loss.item())
            
            #params.trainingList.append(loss.cpu().detach().numpy())

            cumLoss += loss


            if (i+1)%batchRollout == 0:
                cumLoss.backward()
                optimizer.step()

                if debug:
                    max_grad_value = max(torch.max(torch.abs(param.grad)).item() for param in model.parameters() if param.grad is not None)
                    #wandb.log({'max_gradVal':max_grad_value})

                cumLoss = 0
                



            if debug:

                if ((i+1) % 25 == 0 or i == 0):
                
                    max_param_value = max(torch.max(torch.abs(param)).item() for param in model.parameters())
                    #wandb.log({'step': i, 'epoch':epoch, 'Max_paramVal':max_param_value, 'loss pred': loss0, 'L1 Reg':loss1})
                    


            if (((i+1) % 75000) == 0 or i == 0):
                model.eval()
                with torch.no_grad():
                    evalLoss, evalLossSim = evaluate(model, params, device = device)
                    #wandb.log({'step': i,'eval_loss': evalLoss, 'sim loss': evalLossSim})
                model.train()
                
                print(f'EvalLoss -> {evalLoss}')
                print(f'EvalLoss -> {evalLossSim}')
                                     
                                     
                list_eval.append(evalLoss.item())
                list_evalSim.append(evalLossSim.item())
                
                path_save_loss = get_name_file(path_result_MIN, 'loss_list.npy')
                path_save_eval_loss = get_name_file(path_result_MIN, 'eval_list.npy')
                path_save_eval_sim_loss = get_name_file(path_result_MIN, 'eval_sim_list.npy')
                
                data_list_loss = mergeData(path_save_loss, np.array(list_loss))
                data_list_eval_loss = mergeData(path_save_eval_loss, np.array(list_eval))
                data_list_eval_sim_loss = mergeData(path_save_eval_sim_loss, np.array(list_evalSim))
                
                save_add(path_save_loss, data_list_loss, out_mode = 'numpy')
                save_add(path_save_eval_loss, data_list_eval_loss, out_mode = 'numpy')
                save_add(path_save_eval_sim_loss, data_list_eval_sim_loss, out_mode = 'numpy')
                
                list_loss = []
                list_eval = []
                list_evalSim = []


            if (i % 100000) == 0:
                torch.save(model.state_dict(), pathSave)
                
                
                name_checkt_save = f'{cfg_save}_step-{i}.pth'
                save_checkpoint(i, model, optimizer, schedule.scheduler, name_checkt_save)

            
            i += 1
            

    return model