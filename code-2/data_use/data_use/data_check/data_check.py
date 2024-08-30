import torch
import numpy as np
import os
import re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.utils import degree as deg


DATASETS = ['mew_0.001_normal',
            'mew_0.001_noisy',
            'mew_0.01_normal',
            'mew_0.01_noisy',
            'mew_0.001_normal_v2',
            'mew_0.001_noisy_v2',
            'mew_0.01_normal_v2',
            'mew_0.01_noisy_v2',
            ]

def prepare_path(dataset):
    torch_dataset = f'/scratch/users/jpierre/{dataset}/training/torch_file'
    np_dataset = f'/scratch/users/jpierre/{dataset}/training/np_file'
    df_file = f'/scratch/users/jpierre/{dataset}'

    p_out = os.path.join(os.getcwd(), f'{dataset}')
    if not os.path.exists(p_out):
        os.makedirs(p_out)

    return torch_dataset, np_dataset, df_file, p_out



def getSimulations(file, nb = 10, nbSteps = 200):
    
    sim_numbers = set()
    pattern = r"sim_(\d+)_step_"
    for filename in os.listdir(file):
        match = re.search(pattern, filename)
        if match:
            sim_number = match.group(1)
            sim_numbers.add(int(sim_number))
            
    v = sorted(sim_numbers)
        
    random.shuffle(v)
    
    vals = v[:nb]
    nameList = []
    
    
    for v in vals:
        for i in range(nbSteps):
            name = f'{file}/sim_{v}_step_{i}.pt'
            
            nameList.append(name)
            
    return nameList, v



def getSimulation(torch_directory, simNb):
    
    vals = []
    
    p = f'{torch_directory}/sim_{simNb}'
    for filename in os.listdir(torch_directory):
        if filename.startswith(p):
            vals.append(filename)
            
    return vals


def plotDistr(v, bins = 'auto', density = True, xlabel = 'x', ylabel = 'Density', nameFile = None):
    plt.hist(v, bins = bins, density = density)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if nameFile:
        plt.savefig(nameFile)
        plt.close()
    else:
        plt.show()


def calculate_distances(graph):
    start_points = graph.x[graph.edge_index[0]]
    end_points = graph.x[graph.edge_index[1]]

    # Calculate the Euclidean distances
    distances = torch.sqrt(torch.sum((start_points - end_points) ** 2, dim=1))
    return distances


def main():
    for i in tqdm(range(len(DATASETS))):
        dataset = DATASETS[i]

        torch_dataset, np_dataset, df_file, p_out = prepare_path(dataset)

        nameList, nbList = getSimulations(torch_dataset)
        data = [torch.load(f) for f in nameList]

        # speed part

        speeds = [data[i].y[:, :, 0] for i in tqdm(range(len(data)))]
        s = torch.stack(speeds)

        # s_x

        speed_x = s[:, :, 0].view(-1).numpy()
        plotDistr(speed_x, bins = 100, xlabel = 'speed_x', nameFile = f'{p_out}/speed_x.png')

        # s_y

        speed_y = s[:, :, 1].view(-1).numpy()
        plotDistr(speed_y, bins = 100, xlabel = 'speed_y', nameFile = f'{p_out}/speed_y.png')

        speed = torch.sqrt(s[:, :, 0].view(-1) ** 2 + s[:, :, 1].view(-1) ** 2).numpy()
        plotDistr(speed, bins = 100, xlabel = 'speed norm', nameFile = f'{p_out}/speed_norm.png')


        # degree part

        degs = torch.stack([deg(data[i].edge_index[0, :], num_nodes=data[i].x.size(0)) for i in range(len(data))]).view(-1).numpy()

        plotDistr(degs, bins = 'auto', density = False, xlabel = 'Degree', ylabel = 'Number of instances', nameFile = f'{p_out}/degree_hist.png')

        # distance part

        dist = []

        for i in range(len(data)):
            dist.extend(calculate_distances(data[i]).numpy().tolist())
            
        dist = np.array(dist)

        plotDistr(dist, bins = 100, xlabel = 'Distances', nameFile = f'{p_out}/dist_hist.png')


if __name__ == '__main__':
    main()