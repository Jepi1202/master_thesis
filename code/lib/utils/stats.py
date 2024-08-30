import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree as deg


def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

import simulation_v2 as sim
import features as ft
import utils.loading as load
import utils.testing_gen as gen
from utils.tools import array2List, makedirs, writeJson
import utils.nn_gen as nn_gen

import pandas as pd
import matplotlib.ticker as ticker

import simulation_v2 as sim
import features as ft
import utils.loading as load
import utils.testing_gen as gen
from utils.tools import array2List, makedirs, writeJson
import utils.nn_gen as nn_gen




DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


#PATH = 'master/code/runs1'
#PATH = 'master/code/runs2'
#PATH = ['/master/code/analyze_models/exps/test_new_activation_0']
PATH = ['/master/code/results/models/normal/0-01/v']

#DISPLAY_PATH = 'master/code/display_l1'
#DISPLAY_PATH = '/master/code/display_l1_2'
#DISPLAY_PATH = ['/master/code/analyze_models/display/test_new_activation_0']
DISPLAY_PATH = [os.path.join(os.getcwd(), 'figures')]
print(DISPLAY_PATH)

MODEL_PATH = '/master/code/models'

NB_SIM = 5



class ID():
    def __init__(self):
        self.features_x = None
        self.features_edge = None
        self.MLP_hidden = None
        self.l1 = None
        self.dropout = None
        self.layer_norm = None
        self.nb_layer = None
        self.model = None
        self.dt = None

        self.data_type = None


    def get_name(self):
        name = f'{self.model}_'

        if self.data_type:
            name += f'dType-{self.data_type}_'

        if self.features_x:
            name += f'featX-{self.features_x}_'

        if self.features_edge:
            name += f'featE-{self.features_edge}_'

        if self.MLP_hidden:
            name += f'nbHiddenMLP-{self.MLP_hidden}_'

        if self.l1:
            name += f'l1-{self.l1}_'

        if self.dropout:
            name += f'dropout-{self.dropout}_'

        if self.layer_norm:
            name += f'layerNorm-{self.layer_norm}_'

        if self.nb_layer:
            name += f'nbLayer-{self.nb_layer}_'

        if self.dt:
            name += f'dt-{self.dt}_'


        if name.endswith('_'):
            name = name[:-1]

        return name
    

    def load_from_wb(self, name):

        splits = name.split('_')

        self.model = 'simplest'

        if 'baseline' in name:
            self.model = 'baseline'

        if 'complex' in name:
            print(name)
            self.model = 'gnn' 

        if 'gat' in name:
            self.model = 'gat'


        #########

        if 'noisy' in name:
            self.data_type = 'noisy'

        elif 'normal' in name:
            self.data_type = 'normal'

        for s in splits:
            if 'featX' in s:
                self.features_x =s.split('-')[-1]
            if 'featE' in s:
                self.features_edge = s.split('-')[-1]
            if 'nbHiddenMLP' in s:
                self.MLP_hidden = int(s.split('-')[-1])
            if 'l1' in s:
                self.l1 = float(s.split('-')[-1])
            if 'dropout' in s:
                self.dropout = s.split('-')[0]
            if 'layerNorm' in s:
                self.layer_norm = s.split('-')[-1]
            if 'nbLayer' in s:
                self.nb_layer = int(s.split('-')[-1])
            if 'dt' in s:
                self.dt = float(s.split('-')[-1])


            if 'dType' in s:
                self.features_x =s.split('-')[-1]



        



def getParams():
    params = gen.Parameters_Simulation()  


    params.dt = 0.01
    params.v0 = 60
    params.k = 70
    params.epsilon = 0.5
    params.tau = 3.5
    params.R = 1
    params.N = 200
    params.boundary = 100
    params.nbStep = 300


    params.noisy = 0        # function dans utils
    params.features_x = 'v'
    params.features_edge = 'first'


    return params



def find_models_and_paths(model_path:str, display_path:str)->tuple:
    model_list = []
    out_list = []
    ids_list = []
    
    nb = 0

    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.pt'):
                f = os.path.join(root, file)
                model_list.append(f)

                # add ID detection here

                id = ID()

                id.load_from_wb(f.split('/')[-3])

                path_d = id.get_name()

                ids_list.append(id)
                

                #f_display = makedirs(display_path)

                out_list.append(path_d)
                nb += 1


    return model_list, out_list, ids_list



def get_gt_preds(model, graphs, device = DEVICE):

    list_gt = []
    list_preds = []

    with torch.no_grad():
    
        for graph in graphs:

            graph = graph.to(device)
            list_gt.append(graph.y[:, 0, :].cpu().detach().numpy())

            list_preds.append(model(graph).cpu().detach().numpy())

    return list_gt, list_preds



def plotBoxPlot(diff, vals, bins, xlabel = 'X values', ylabel = 'y values', showfliers = False):
    

    groups = np.digitize(diff, bins)
    grouped_errors = {i: [] for i in range(len(bins)+1)}

    for idx, group in enumerate(groups):
        grouped_errors[group].append(vals[idx])


    centers = [bins[0]-1] + [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)] + [bins[-1] + 1]
    data_to_plot = [grouped_errors[k] for k in sorted(grouped_errors.keys())]
    medians = [np.median(g) if g else np.nan for g in data_to_plot]

    boxprops = dict(linestyle='-', linewidth=2, color='black')  # Custom box properties
    medianprops = dict(linestyle='-', linewidth=0, color='orange')  # Invisible median line
    boxplot_elements = plt.boxplot(data_to_plot, positions=centers, boxprops=boxprops, medianprops=medianprops, showfliers=showfliers, widths = 0.2)

    plt.plot(centers, medians, 'o-', color='orange', label='Medians')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return medians


########################################


def errorsDiv(pred, gt):
    
    v = np.pi * 2
    anglePred = np.arctan2(pred[:, 1], pred[:, 0])
    angleGT = np.arctan2(gt[:, 1], gt[:, 0])

    normPred = np.linalg.norm(pred, axis=-1)
    normGT = np.linalg.norm(gt, axis=-1)

    errorNorm = np.abs(normPred - normGT)
    errorAngle = np.abs((anglePred - angleGT + np.pi) % (2 * np.pi) - np.pi)

    return errorAngle, errorNorm




########################################


def getStdMessage(model, attr):
    model.eval()
    with torch.no_grad():
        v = model.GNN.message(None, None, attr).cpu().detach().numpy()
    return v



def plotStdMessage(messages):

    std = np.std(messages, axis = 0)

    plt.plot(std)
    plt.xlabel('Features')
    plt.ylabel('Standard Deviation')

    return std


########################################


def plotDegreeLoss(degreeList, model, graphs, device = DEVICE):

    list_gt, list_preds = get_gt_preds(model, graphs, device = device)


    errors_MSE = []

    for i in range(len(list_gt)):
        e = np.mean(get_ma(list_preds[i], list_gt[i]), axis = -1)
        
        errors_MSE.append(e)

    errors_MSE = np.stack(errors_MSE).reshape(-1)


    degList = [degreeList[i].cpu().detach().numpy() for i in range(len(degreeList))]
    degList = np.stack(degList).reshape(-1)

    maxBin = min(np.max(degList), 8)
    bins = np.arange(maxBin+2) - 0.5

    plotBoxPlot(degList, errors_MSE, bins, xlabel = 'Degree', ylabel = 'L1 Error')

    return None


def get_degree_list(grpahs):

    deg_list = []

    for graph in grpahs:
        degs = deg(graph.edge_index[0, :], num_nodes=graph.x.size(0))
        deg_list.append(degs)

    return deg_list



################################################


def plotDistLoss(distList, errorList):
    bins = np.linspace(0, 6, 20)

    plotBoxPlot(distList, errorList, bins, xlabel = 'Distance', ylabel = 'L1 Error')

    return None


###############################################

def get_se(gts, preds):

    vals = (gts - preds) ** 2

    return vals

def get_ma(gts, preds):

    vals = np.abs(gts - preds)

    return vals


def get_mae(gts, preds):

    vals = np.abs(gts - preds)

    return np.mean(vals), np.std(vals)


###############################################


def error_speed(model, graphs):

    speed_norm = []
    errors = []

    for graph in graphs:
        speeds = graph.x[:, 2:4]
        norm = np.linalg.norm(speeds.cpu().detach().numpy(), axis= -1).reshape(-1)
        speed_norm.extend(norm.tolist())

        preds = model(graph).cpu().detach().numpy()
        y = graph.y.cpu().detach().numpy()[:, 0, :]

        ers = np.mean((np.abs(preds - y)), axis = -1)      ## MAE here



        errors.extend(ers.tolist())

    speed_norm = np.array(speed_norm)
    errors = np.array(errors)

    print(errors.shape)
    print(speed_norm.shape)

    plt.scatter(speed_norm, errors)
    plt.grid()
    plt.xlabel('Speed norm')
    plt.ylabel('L1 Error')


    return speed_norm, errors




############################################################

def get_pd_dict(d: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(d)
    # keys are columns by default (^^)

    return df





def plotBoxPlot2(diff, vals, bins, xlabel = 'X values', ylabel = 'y values', showfliers = False):
    

    groups = np.digitize(diff, bins)
    grouped_errors = {i: [] for i in range(len(bins)+1)}

    for idx, group in enumerate(groups):
        grouped_errors[group].append(vals[idx])

    plt.grid(zorder = 2)
    centers = [bins[0]-1] + [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)] + [bins[-1] + 1]
    data_to_plot = [grouped_errors[k] for k in sorted(grouped_errors.keys())]
    medians = [np.median(g) if g else np.nan for g in data_to_plot]

    boxprops = dict(linestyle='-', linewidth=1, color='black')  # Custom box properties
    medianprops = dict(linestyle='-', linewidth=0, color='orange')  # Invisible median line
    boxplot_elements = plt.boxplot(data_to_plot, positions=centers, boxprops=boxprops, medianprops=medianprops, showfliers=showfliers, widths = 0.0051, zorder = 1)


    plt.plot(centers[:-1], medians[:-1], '-', color='orange', label='Medians', alpha = 0.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #v = np.linalg.norm(0, 0.15, 5)
    #plt.xticks(ticks=v, labels=[f'{v[i]}' for i in len(v)])

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f'{val:.2g}'))
    
    plt.xlim([-0.001, np.max(bins) + 0.01])

    return medians

        
##########################################################



def getData(params, nbSim = NB_SIM):
    data_gt = gen.get_mult_data(params, nbSim)
    graphs_gt = gen.sims2Graphs(data_gt, params.features_x)

    return graphs_gt, data_gt

########################
########################
########################
########################
########################
########################



########################################

def MSE_rollout(roll, sim, display:bool = False):
    x = np.arange(roll.shape[0])
    vals = ((roll - sim) ** 2).reshape(x.shape[0], -1)
    y = np.mean(vals, axis=1)
    std = np.std(vals, axis=1)

    if display:
        plt.plot(x, y, 'blue')
        plt.fill_between(x, y-std, y+std, zorder=  2, alpha = 0.4)
        plt.xlabel('Time')
        plt.ylabel('Rollout MSE')
        plt.grid(zorder = 1)

    return x, y


def applyMSE_rollout(simList, predList, color = 'blue', display:bool = True):

    res = np.zeros((len(simList), simList[0].shape[0]))

    for i in range(len(simList)):
        _, res[i] = MSE_rollout(predList[i], simList[i])

    res = np.mean(res, axis = 0)
    std_MSE = np.std(res, axis = 0)

    x = np.arange(res.shape[0])
    if display:
        plt.plot(x, res, color)
        #std = np.std(res, axis = 0)
        #plt.fill_between(x, res-std, res+std, zorder=  2, alpha = 0.4)
        plt.xlabel('Timesteps')
        plt.ylabel('Rollout MSE')
        plt.grid(zorder = 1)
        plt.grid()


    
    return res, std_MSE


###############################################

def MSD_comp(traj, tau):
    T = traj.shape[0]
    i = np.arange(T - tau)
    j = i + tau

    #return np.linalg.norm(traj[j, :, :] - traj[i, :, :], axis=-1)**2
    return (traj[j, :, :] - traj[i, :, :])**2


def MSD(traj: np.array)-> np.array:
    """
    Allows to compute the Mean Squared Displacement of the trajectories for all timestamps
    
    Args:
    -----
    - `traj`: np.array of N trajectories of length T [NxT]
    
    Output:
    -------
    Mean Squared Displacement for all timestamps
    """

    res = []
    T = traj.shape[0]
    
    for tau in range(1, T):
        val = np.mean(np.mean(MSD_comp(traj, tau), axis=0), axis=0)
        res.append(val)

    return res

def applyMSD(sims:list, display:bool = False, color:str = 'blue')->np.array:
    """ 
    Function to apply MSD to a group of simulations

    NOTE: test

    Args:
    -----
        - `sims` (list): list of simualtions

    Returns:
    --------
        np array [#Sim, T-1] of MSD computations
    """

    res = np.zeros((len(sims), sims[0].shape[0]-1))
    for i in range(len(sims)):
        sim = sims[i]
        res[i, :] = np.array(MSD(sim))


    if display:
        x = np.arange(sims[0].shape[0]-1)
        y = np.mean(res, axis = 0)
        std = np.std(res, axis = 0)
        plt.loglog(x, y, color = color, zorder = 1)
        plt.fill_between((x), (y - std), (y+std), color = color, alpha = 0.4, zorder = 2)
        
        plt.xlabel('Timesteps')
        plt.ylabel('MSD')
        plt.grid()

    return res


################################################


def SelfIntermediateA(data, qval, verbose=False):
    T, N, _ = data.shape  # T is the number of time steps, N is the number of particles
    qval = np.array(qval, dtype=np.complex128)  # Ensure wave vector is complex for the computation
    
    SelfInt = np.empty((T-1,), dtype=np.complex128)
    
    for t in range(T-1):
        smax = T - t  
        
        rt = data[:smax]       
        rtplus = data[t:] 
        dr = rt - rtplus
        
        exp_factor = np.exp(1j * (qval[0] * dr[:, :, 0] + qval[1] * dr[:, :, 1]))
        
        SelfInt[t] = np.sum(exp_factor) / (N * smax)
    
    SelfInt_mod = np.sqrt(np.real(SelfInt)**2 + np.imag(SelfInt)**2)
    
    tval = np.linspace(0, T-1, num=T-1)
    
    if verbose:
        plt.figure(figsize=(10, 5))
        plt.semilogy(tval, SelfInt_mod, '.-r', lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)
        plt.ylim([0, 1])
        plt.show()
    
    return tval, SelfInt_mod, SelfInt


def applySelfScattering(simList, qval = None, display:bool = False, color = 'blue'):
    if qval is None:
        R = 1       #ngfjdngjdnsjhnjfnvjfdnbjvnfdjkgnfjkdhvbjkdhgjklhgjkd
        qval = 2*np.pi/R*np.array([1,0])

    res = np.zeros((len(simList),simList[0].shape[0]-1 ))

    for i in range(len(simList)):
        sim = simList[i]

        val = SelfIntermediateA(sim.copy(), qval.copy(), verbose = False)[1]
        res[i] = val

    r = np.mean(res, axis=0)
    delta = np.std(r, axis = 0)

    if display:
        #plt.figure(figsize=(10, 5))
        t = np.arange(1, simList[0].shape[0])

        plt.grid()
        plt.semilogy(t, r, color = color, lw=2)
       # plt.fill_between(t, r-delta, r+delta, color = color, alpha = 0.4)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate Function')
        plt.grid(True)

    return r
        

def apply_mean_speed(gt_sim, pred_sim):
    inds = np.arange(gt_sim.shape[1]-1)
    inds2 = inds + 1

    speeds_gt = np.linalg.norm(gt_sim[:, inds2 - inds, :, :], axis = -1)

    speeds_preds = np.linalg.norm(pred_sim[:, inds2 - inds, :, :], axis = -1)


    mean_s_gt = np.mean(speeds_gt, axis = -1)
    std_s_gt = np.std(speeds_gt, axis = -1)

    mean_s_preds = np.mean(speeds_preds, axis = -1)
    std_s_preds = np.std(speeds_preds, axis = -1)


    mean_s_gt = np.mean(mean_s_gt, axis = 0)
    std_s_gt = np.mean(std_s_gt, axis = 0)
    mean_s_preds = np.mean(mean_s_preds, axis = 0)
    std_s_preds = np.mean(std_s_preds, axis = 0)


    plt.plot(mean_s_gt, 'g', label = 'Ground truth')
    plt.fill_between(inds, mean_s_gt - std_s_gt,  mean_s_gt + std_s_gt, alpha = 0.4)

    plt.plot(mean_s_gt, 'g', label = 'Predictions')
    plt.fill_between(inds, mean_s_preds - std_s_preds,  mean_s_preds + std_s_preds, alpha = 0.4)

    plt.grid()
    plt.xlabel('Timesteps')

    plt.ylabel('Speed [micro meters / hour]')




def run_bash_ev(model, data_gt, file, data, id, nbStep = 100, params = None):
    # data_gt[n_sim, T, N, 2]
    # data= dict

    ## compute the predictions

    print(id.__dict__)

    if id.features_x == 'v':
        preds = nn_gen.generate_sim_batch(model, data_gt, dt_scale = params.dt)
        data_preds_list = array2List(preds)


    else:
        preds = nn_gen.generate_sim_batch(model, data_gt)
        data_preds_list = array2List(preds)
    
    ## adjust the shape of the ground truth

    data_gt_list = array2List(data_gt)
    data_gt = data_gt[:, :nbStep, :, :]
    data_gt_list = [d[:nbStep, :, :] for d in data_gt_list]


    #if data_preds_list[0].shape[0] != data_gt_list[0].shape[0]:
    preds = preds[:, :nbStep, :, :]
    data_preds_list = [d[:nbStep, :, :] for d in data_preds_list]



    print(data_preds_list[0].shape)
    print(data_gt_list[0].shape)



    res_mse, std_mse = applyMSE_rollout(data_gt_list, data_preds_list)
    plt.savefig(os.path.join(file, 'mse_rollout.png'))
    plt.close()

    data['MSE_rollout'] = np.mean(res_mse)
    data['MSE_rollout_std'] = np.mean(std_mse)

    writeJson(data, os.path.join(file, 'data2.json'))


    r = applyMSD(data_gt_list,
                            display = True)
    

    np.save(os.path.join(file, 'MSD_gt.json'), r)

    r = applyMSD(data_preds_list,
            display = True,
            color='red')
    
    np.save(os.path.join(file, 'MSD_pred.json'), r)
    
    plt.savefig(os.path.join(file, 'msd.png'))
    plt.close()



    r = applySelfScattering(data_gt_list, 
            qval = None, 
            display = True, 
            color = 'blue')


    r = applySelfScattering(data_preds_list, 
                        qval = None, 
                        display = True, 
                        color = 'red')
    
    plt.savefig(os.path.join(file, 'scattering.png'))
    plt.close()



    apply_mean_speed(data_gt, preds)
    plt.savefig(os.path.join(file, 'mean-speeds.png'))
    plt.close()


    return data





def perform_1_step_stats(data_gt, graphs_gt, list_exp, list_disp, params, batch = True, nbStep = 100):


    # praamters of the silualtiosn

    
    #graphs_gt = getData(params)

    
    for i in range(len(list_exp)):

        models_dir = list_exp[i]
        path_dis = list_disp[i]

        models, paths, id_list = find_models_and_paths(models_dir, path_dis)


        for j in range(len(models)):

            model_p = models[j]
            path = paths[j]
            id = id_list[j]

            path_display_exp = makedirs(os.path.join(path_dis, path))
        
            model = load.loadModel(load.getModelName(model_p), path=MODEL_PATH)
            std_dict = torch.load(model_p, map_location = 'cpu')
            model.load_state_dict(std_dict)
            model.eval()
            model = model.to(DEVICE)


            ## compute the predictions


            gt, preds = get_gt_preds(model, graphs_gt)

            # complete the dataframe

            # angles
            error_norm = []
            error_angle = []
            for i in range(len(preds)):
                e_norm, e_angle = errorsDiv(preds[i], gt[i])

                error_norm.append(e_norm)
                error_angle.append(e_angle)

            error_angle = np.stack(error_angle).reshape(-1)
            error_norm = np.stack(error_norm).reshape(-1)
            
            # mae

            errors_MAE = []

            for i in range(len(preds)):
                e = get_ma(preds[i], gt[i])
                errors_MAE.append(e)

            errors_MAE = np.stack(errors_MAE).reshape(-1)


            ### values + std ...


            # save json

            d = {}

            d['MAE'] = float(np.mean(errors_MAE))
            d['MAE-std'] = float(np.std(errors_MAE))

            d['angle-error'] =float(np.median(error_angle))
            d['angle-std'] = float(np.std(error_angle))

            d['norm-error'] = float(np.median(error_norm))
            d['norm-std'] = float(np.std(error_norm))

            writeJson(d, os.path.join(path_display_exp, 'data.json'))
            print(os.path.join(path_display_exp, 'data.json'))


            ####### compute the stats

            # messages

            messages = getStdMessage(model, graphs_gt[0].edge_attr.to(DEVICE))
            _ =  plotStdMessage(messages)
            plt.savefig(os.path.join(path_display_exp, 'messages.png'))
            plt.close()


            # degrees

            deg_list = get_degree_list(graphs_gt)

            plotDegreeLoss(deg_list, model, graphs_gt)
            plt.savefig(os.path.join(path_display_exp, 'errors-degs.png'))
            plt.close()


            # scatter plot MSE - speed norm


            speed_norm, errors = error_speed(model, graphs_gt)
            plt.savefig(os.path.join(path_display_exp, 'mae-speed-norm-scatter.png'))
            plt.close()


            maxBin = 20
            bins = np.linspace(0, np.max(speed_norm), maxBin+2)


            _ = plotBoxPlot2(speed_norm, errors, bins, xlabel = 'Absolute speed', ylabel = 'L1 Error')
            plt.savefig(os.path.join(path_display_exp, 'mae-speed-norm-boxplots.png'))
            plt.close()


            if batch:
                run_bash_ev(model, data_gt, path_display_exp, d, nbStep=nbStep, id=id, params=params)






