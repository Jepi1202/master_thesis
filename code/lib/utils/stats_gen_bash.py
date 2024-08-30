import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm
import os
import torch
from torch_geometric.data import Data

def path_link(path:str):
    sys.path.append(path)

path_link('/master/code/lib')

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
PATH = ['/master/code/analyze_models/exps/exp-test']

#DISPLAY_PATH = 'master/code/display_l1'
#DISPLAY_PATH = '/master/code/display_l1_2'
#DISPLAY_PATH = ['/master/code/analyze_models/display/test_new_activation_0']
DISPLAY_PATH = ['/master/code/analyze_models/display/exp-test']

MODEL_PATH = '/master/code/models/mod_base'

NB_SIM = 10


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


def applyMSE_rollout(simList, predList, color = 'blue', display:bool = False):

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

    return np.linalg.norm(traj[j, :, :] - traj[i, :, :], axis=-1)**2


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
        




def run_bash_ev(model, data_gt, file, data):
    ## compute the predictions

    preds = nn_gen.generate_sim_batch(model, data_gt)
    data_preds_list = array2List(preds)
    
    ## adjust the shape of the ground truth

    data_gt = data_gt[:, :preds.shape[1], :, :]
    data_gt_list = [d[:preds.shape[1], :, :] for d in data_gt_list]



    res_mse, std_mse = applyMSE_rollout(data_gt_list, data_preds_list)
    plt.savefig(os.path.join(file, 'mse_rollout.png'))
    plt.close()

    data['MSE_rollout'] = np.mean(res_mse)
    data['MSE_rollout_std'] = np.mean(std_mse)

    writeJson(data, os.path.join(file, 'data2.json'))


    r = applyMSD(data_gt_list,
                            display = True)

    r = applyMSD(data_preds_list,
            display = True,
            color='red')
    
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



    return data









def main():

    if PATH is None:
        # check if listdir only outputs the last element (...)
        #list_exp = [os.listdir('/master/code/analyze_models/exp/')]
        #list_disp = [os.path.join('/master/code/analyze_models/display', list_exp[i]) for i in range(len(list_exp))]
        print('jfbgjksdfj;gnvdfjksvnkjdfbkjsdfb v,df')


    else:
        list_exp = PATH                 # folder with many nn
        list_disp = DISPLAY_PATH


    # get gt

    params = gen.Parameters_Simulation()
    data_gt = gen.get_mult_data(params, NB_SIM)
    data_gt_list = array2List(data_gt)

    
    for i in range(len(list_exp)):

        exp = list_exp[i]
        disp = list_disp[i]

        model_list = load.findModels(exp)

        for model_path in model_list:
            name_model = load.getName(model_path)
        
            model = load.loadModel(load.getModelName(name_model), path=MODEL_PATH)
            std_dict = torch.load(model_path)
            model.load_state_dict(std_dict)
            model.eval()
            model = model.to(DEVICE)


            ## compute the predictions

            preds = nn_gen.generate_sim_batch(model, data_gt)
            data_preds = array2List(preds)
            
            ## adjust the shape of the ground truth

            data_gt = data_gt[:, :preds.shape[1], :, :]
            data_gt_list = [d[:preds.shape[1], :, :] for d in data_gt_list]

            ## create the folder

            f = makedirs(os.path.join(os.getcwd(), 'figures'))

            # perform the stats here

            _ = applyMSE_rollout(data_gt_list, data_preds)
            plt.savefig(os.path.join(f, 'mse_rollout.png'))
            plt.close()


            r = applyMSD(data_gt_list,
                            display = True)

            r = applyMSD(data_preds,
                    display = True,
                    color='red')
            
            plt.savefig(os.path.join(f, 'msd.png'))
            plt.close()



            r = applySelfScattering(data_gt_list, 
                    qval = None, 
                    display = True, 
                    color = 'blue')


            r = applySelfScattering(data_preds, 
                                qval = None, 
                                display = True, 
                                color = 'red')
            
            plt.savefig(os.path.join(f, 'scattering.png'))
            plt.close()











if __name__ == '__main__':
    main()
    