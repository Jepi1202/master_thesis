import torch
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.cuda.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Param_eval():
    def __init__(self, path = None, wandbName = None):
        self.path = path
        self.wandbName = wandbName

class EvaluationCfg():
    def __init__(self):

        self.jacobian = None

        self.norm_angleError = None

        self.heatmap = None

        self.L1_vect = None

        self.MSE_rollout = None

        self.degree_error = None



def evaluateLoad(loader, cfg: EvaluationCfg, device = DEVICE):

    evalLoss = 0
    nbCall = len(loader)

    res = {}

    distList = []           # list of distances
    errorList = []          # list of errors
    degreeList = []         # list of degrees
    normError = []          # list of norm errors
    angleError = []         # list of angles errors
    messages = None
    
    for d, _ in loader:
        d = d.to(device)
        d.x = d.x[:, 2:]
        d = normalizeGraph(d)
        pred = model(d)  
        
        d.y = torch.swapaxes(d.y, 0, 1)
        

        evalLoss += torch.nn.functional.l1_loss(pred.reshape(-1), d.y[0, :, :].reshape(-1))


        if cfg.jacobian:
            pass


        if cfg.degree_error:
            degs = deg(d.edge_index[0, :], num_nodes=d.x.size(0))
            errorList.extend(errors.cpu().numpy().tolist())
            degreeList.extend(degs.cpu().numpy().tolist())


        if cfg.dist_error:
            distList.extend(dist.cpu().numpy().tolist()) 
            if not cfg.degree_error:
                errorList.extend(errors.cpu().numpy().tolist())


        if cfg.norm_angleError:
            
            errorAngle, errorNorm = errorsDiv(pred.cpu().numpy(), d.y[0, :, :].cpu().numpy())
            normError.extend(errorNorm.tolist())
            angleError.extend(errorAngle.tolist())


        if self.L1_vect:
            m = getStdMessage(mod, s.edge_attr)
            if messages is None:
                messages = m
            else:
                messages = np.vstack((messages, m))


    # saving results

    evalLoss = evalLoss / nbCall

    if cfg.dist_error:
        distList = np.array(distList)
        res['distList'] = distList
    
    if cfg.dist_error or cfg.degree_error:
        errorList = np.array(errorList)
        res['errorList'] = errorList

    if cfg.degree_error:
        degreeList = np.array(degreeList)
        res['degreeList'] = degreeList

    if cfg.norm_angleError:
        normError = np.array(normError)
        angleError = np.array(angleError)
        res['angleError'] = angleError
        res['normError'] = normError


    res['evalLoss'] = evalLoss

    return res


def evaluateSim(loader, cfg, device = DEVICE):
    
    res = {}
    data_x = None
    data_y = None

    nbRoll = 80
    nbCalls = len(loader)

    evalLossSim = 0
    mse = 0

    for d, _ in loader:
                    
        d = torch.squeeze(d, dim = 0).numpy()
        start = 8       # not 0
        res = getSimulationData(model, nbRoll, d, i = start)

        if data_x is None:
            data_x = res.numpy()
        else:
            data_x = np.concatenate((data_x, res.numpy()), axis = 0)

        if data_y is None:
            data_y = d
        else:
            data_y = np.concatenate((data_y, d), axis = 0)


        L = res.shape[0]
                        
        evalLossSim += torch.nn.functional.l1_loss(res.reshape(-1), torch.from_numpy(d[start:(start + L), :, :].copy()).reshape(-1).to(device))

        if cfg.MSE_rollout:
            mse += MSE_rollout(res.copy(), d)

    res['evalLossSim'] = evalLossSim / nbCalls

    if cfg.MSE_rollout:
        res['mse_rollout'] = mse / nbCalls


    return res

    
def compute_jacobian(model, inputs) -> torch.Tensor:
    """
    Compute the jacobian matrix with respect to the inputs
    
    Args:
    -----
        - `model` (torch.nn.Module): The neural network model.
        - `inputs` (torch.Tensor): The inputs to the neural network.
    
    Returns:
    --------
        tuple of the jacobians
    """
    model.eval()
    inputs.requires_grad_(True)
    
    outputs = model(inputs)
    jacobian1 = torch.zeros_like(inputs)
    jacobian2 = torch.zeros_like(inputs)


    for i in range(outputs.shape[0]):
        model.zero_grad()
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i, 0] = 1  
        outputs.backward(gradient=grad_outputs, retain_graph=True)
        jacobian1[i] = inputs.grad.detach().clone()
        inputs.grad.zero_()



    for i in range(outputs.shape[0]):
        model.zero_grad()
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i, 1] = 1  # Set up for the second output dimension
        outputs.backward(gradient=grad_outputs, retain_graph=True)
        jacobian2[i] = inputs.grad.detach().clone()
        inputs.grad.zero_()
    
    
    # Return the Jacobian matrix
    return jacobian1, jacobian2




def compute_graph_jacobian(model, inputs) -> torch.Tensor:
    """
    Compute the jacobian matrix with respect to the inputs
    
    Args:
    -----
        - `model` (torch.nn.Module): The neural network model.
        - `inputs` (torch.Tensor): The inputs to the neural network.
    
    Returns:
    --------
        tuple of the jacobians
    """
    model.eval()

    # x
    inputs.x.requires_grad_(True)
    inputs.edge_attr.requires_grad_(True)
    
    outputs = model(inputs)
    jacobian1 = torch.zeros_like(inputs.x)
    jacobian2 = torch.zeros_like(inputs.x)

    jacobian1edge = torch.zeros_like(inputs.edge_attr)
    jacobian2edge = torch.zeros_like(inputs.edge_attr)


    for i in range(outputs.shape[0]):
        model.zero_grad()
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i, 0] = 1  
        outputs.backward(gradient=grad_outputs, retain_graph=True)
        jacobian1[i] = inputs.x.grad.detach().clone()
        jacobian1edge[i] = inputs.edge_attr.grad.detach().clone()
        inputs.x.grad.zero_()



    for i in range(outputs.shape[0]):
        model.zero_grad()
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i, 1] = 1  # Set up for the second output dimension
        outputs.backward(gradient=grad_outputs, retain_graph=True)
        jacobian2[i] = inputs.grad.detach().clone()
        jacobian2edge[i] = inputs.edge_attr.grad.detach().clone()
        inputs.grad.zero_()
    
    
    # Return the Jacobian matrix
    return jacobian1, jacobian2, jacobian1edge, jacobian2edge



def meanJacobian(model, input):
    j1, j2, jedge1, jedge2 = compute_graph_jacobian(model, input)

    j1 = j1.cpu().numpy()
    j2 = j2.cpu().numpy()
    jedge1 = jedge1.cpu().numpy()
    jedge2 = jedge2.cpu().numpy()


    return np.mean(j1, axis = 0), np.mean(j2, axis = 0), np.mean(jedge1, axis = 0), np.mean(jedge2, axis = 0)


####################################

def heatmap(values, positions, grid_size=(50, 50), plot_size=(8, 6), mode = 'max', display = True):
    """
    Visualizes the mean acceleration at different positions in a heatmap manner.

    Parameters:
    - values: np.array of shape [T, N, 2], representing the acceleration vectors of N cells over T timesteps.
    - positions: np.array of shape [T, N, 2], representing the spatial positions of N cells over T timesteps.
    - grid_size: Tuple representing the dimensions of the grid used to calculate mean values.
    - plot_size: Tuple representing the size of the output plot.
    """
    
    # Flatten the position and acceleration_norms arrays
    flattened_positions = positions.reshape(-1, 2)
    flattened_values = values.flatten()

    # Create a grid
    x_positions, y_positions = flattened_positions[:, 0], flattened_positions[:, 1]
    x_edges = np.linspace(x_positions.min(), x_positions.max(), grid_size[0] + 1)
    y_edges = np.linspace(y_positions.min(), y_positions.max(), grid_size[1] + 1)

    # Digitize the positions to find out which grid cell each belongs to
    x_inds = np.digitize(x_positions, x_edges) - 1
    y_inds = np.digitize(y_positions, y_edges) - 1

    # Accumulate the acceleration norms in their respective grid cells and count the entries
    accumulation_grid = np.zeros(grid_size, dtype=np.float64)
    count_grid = np.zeros(grid_size, dtype=np.int32)

    for x_ind, y_ind, v in zip(x_inds, y_inds, flattened_values):
        if 0 <= x_ind < grid_size[0] and 0 <= y_ind < grid_size[1]:
            if mode == 'mean':
                accumulation_grid[x_ind, y_ind] += v
            elif mode == 'max':
                accumulation_grid[x_ind, y_ind] = max(accumulation_grid[x_ind, y_ind], v)
            elif mode == 'min':
                accumulation_grid[x_ind, y_ind] = min(accumulation_grid[x_ind, y_ind], v)

            count_grid[x_ind, y_ind] += 1


    if display:
        # Avoid division by zero
        if mode == 'mean':
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_acceleration_grid = np.true_divide(accumulation_grid, count_grid)

        else:
            mean_acceleration_grid = accumulation_grid

        if np.any(count_grid == 0):
            mean_acceleration_grid[count_grid == 0] = np.nan  # Set cells with no data to NaN

        # Plotting the heatmap
        plt.figure(figsize=plot_size)
        plt.imshow(mean_acceleration_grid.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', cmap='jet')
        plt.colorbar(label='Error')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Heatmap of Error')




def MSE_rollout(roll, sim):
    x = np.arange(roll.shape[0])
    vals = ((roll - sim) ** 2).reshape(x.shape[0], -1)
    y = np.mean(vals, axis=1)
    std = np.std(vals, axis=1)

    plt.plot(x, y, 'blue')
    plt.fill_between(x, y-std, y+std, zorder=  2)
    plt.xlabel('Time')
    plt.ylabel('Rollout MSE')
    plt.grid(zorder = 1)

    return x, y



################################################


def errorsDiv(pred, gt):
    
    anglePred = np.arctan2(pred[:, 1], pred[:, 0])
    angleGT = np.arctan2(gt[:, 1], gt[:, 0])

    normPred = np.linalg.norm(pred, axis=-1)
    normGT = np.linalg.norm(gt, axis=-1)

    errorNorm = np.abs(normPred - normGT)
    errorAngle = np.abs(anglePred - angleGT)

    return errorAngle, errorNorm



################################################




DEVICE = torch.cuda.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import features as ft
from torch_geometric.data import Data
from norm import normalizeGraph

def oneStepSimulation(sim, model, radius = None, device = DEVICE):

    # get the states

    x, y, resD, resInd = ft.processSimulation(sim,radius=radius)
    res = []
    out = np.zeros_like(sim)
    out[0, :, :] = sim[0, :, :]
    for i in range(len(x)):
        pos = x[i][:, :2].cpu().detach().numpy()
        s = Data(x = x[i][:, 2:], y = y[i], edge_attr = resD[i], edge_index = resInd[i])
        s = normalizeGraph(s)
        s = s.to(device)
        speeds = model(s)
        if (i+1) < out.shape[0]:
            out[i+1, :, :] = pos + speeds

    
    return out


def errorPred(pred, gt):
    
    return np.abs(pred - gt)




################################################




