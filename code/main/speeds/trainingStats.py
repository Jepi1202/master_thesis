import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torch.nn as nn
import os
import sys
import yaml
from torch_geometric.data import Data

from features import getFeatures

with open('cfg.yml', 'r') as file:
    cfg = yaml.safe_load(file) 



OUTPUT_TYPE = cfg['feature']['output']
THRESHOLD_DIST = cfg['feature']['distGraph']
MIN_DIST = cfg['normalization']['distance']['minDistance']
MAX_DIST = cfg['normalization']['distance']['maxDistance']
BOUNDARY = cfg['simulation']['parameters']['boundary']


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    
R = cfg['simulation']['parameters']['R']
MIN_RAD = cfg['normalization']['radius']['minRad']
MAX_RAD = cfg['normalization']['radius']['maxRad']


#pathWeight = '/home/jpierre/v2/part_1/model_trained/model_1_min_2_test_v1.pt'
pathWeight = '/home/jpierre/v2/part_1_b/model_trained/model2_L1_4_min.pt'
pathMod = '/home/jpierre/v2/models/model2.py'



DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


############
# Normalisation
############

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


def heatmap(accelerations, positions, grid_size=(50, 50), plot_size=(8, 6), mode = 'max', display = True):
    """
    Visualizes the mean acceleration at different positions in a heatmap manner.

    Parameters:
    - accelerations: np.array of shape [T, N, 2], representing the acceleration vectors of N cells over T timesteps.
    - positions: np.array of shape [T, N, 2], representing the spatial positions of N cells over T timesteps.
    - grid_size: Tuple representing the dimensions of the grid used to calculate mean accelerations.
    - plot_size: Tuple representing the size of the output plot.
    """
    # Calculate the norm of the acceleration
    acceleration_norms = np.linalg.norm(accelerations, axis=2)
    
    # Flatten the position and acceleration_norms arrays
    flattened_positions = positions.reshape(-1, 2)
    flattened_acceleration_norms = acceleration_norms.flatten()

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

    for x_ind, y_ind, acc_norm in zip(x_inds, y_inds, flattened_acceleration_norms):
        if 0 <= x_ind < grid_size[0] and 0 <= y_ind < grid_size[1]:
            if mode == 'mean':
                accumulation_grid[x_ind, y_ind] += acc_norm
                count_grid[x_ind, y_ind] += 1
            elif mode == 'max':
                accumulation_grid[x_ind, y_ind] = max(accumulation_grid[x_ind, y_ind], acc_norm)
                count_grid[x_ind, y_ind] = 1
            elif mode == 'min':
                accumulation_grid[x_ind, y_ind] = min(accumulation_grid[x_ind, y_ind], acc_norm)
                count_grid[x_ind, y_ind] = 1

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_acceleration_grid = np.true_divide(accumulation_grid, count_grid)
        mean_acceleration_grid[count_grid == 0] = np.nan  # Set cells with no data to NaN

    # Plotting the heatmap
    plt.figure(figsize=plot_size)
    plt.imshow(mean_acceleration_grid.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', cmap='jet')
    plt.colorbar(label='Error')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Heatmap of Error')
    
    if display:
        plt.show()
    
    
    
def create_comparison_video_cv2(data, ground_truth, filename='comparison_simulation.mp4', fps=10, size=(600, 600)):
    """
    Creates an MP4 video from two PyTorch tensors representing predicted and true cell movements, using cv2.
    Cells are colored based on the closeness of their positions to the ground truth.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y) for the predictions.
    - ground_truth: A PyTorch tensor of the same shape as `data`, representing the true positions.
    - filename: Name of the output MP4 file.
    - fps: Frames per second for the output video.
    - size: Size of the output video frame.
    """
    # Convert the data to numpy for easier manipulation
    #data_np = data.cpu().numpy()
    #ground_truth_np = ground_truth.cpu().numpy()
    data_np = data
    ground_truth_np = ground_truth
    
    # Normalize coordinates to fit within the video frame size
    combined = np.concatenate((data_np, ground_truth_np), axis=1)
    min_vals = combined.min(axis=(0, 1), keepdims=True)
    max_vals = combined.max(axis=(0, 1), keepdims=True)
    data_np = (data_np - min_vals) / (max_vals - min_vals) * (np.array([size[0] - 1, size[1] - 1]))
    ground_truth_np = (ground_truth_np - min_vals) / (max_vals - min_vals) * (np.array([size[0] - 1, size[1] - 1]))
    
    # Compute distances and normalize for color mapping
    distances = np.sqrt(np.sum((data_np - ground_truth_np)**2, axis=2))
    normalized_distances = distances / distances.max()
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(filename, fourcc, fps, size)
    
    # Prepare the colormap
    colormap = plt.get_cmap('jet')
    
    for i in range(data_np.shape[0]):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        for (x, y), dist in zip(data_np[i], normalized_distances[i]):
            # Map distance to color
            color = colormap(dist)[:3]  # Get RGB from RGBA
            color = (color[2] * 255, color[1] * 255, color[0] * 255)  # Convert to BGR for OpenCV
            # Draw the cell as a circle
            cv2.circle(frame, (int(x), int(y)), radius=2, color=color, thickness=-1)
        out.write(frame)
    
    out.release()


def optimized_getGraph(mat_t, threshold=THRESHOLD_DIST):
    """
    Optimized function to compute the graph for PyTorch Geometric.

    Args:
    -----
        - `mat_t` (np.array): 2D np array (matrix at a given timestep)
        - `threshold` (float): Distance threshold for connecting vertices

    Returns:
    --------
        - `distList` (torch.Tensor): Tensor of distances and direction cosines.
        - `indices` (torch.Tensor): Tensor of graph indices.
    """
    num_points = mat_t.shape[0]
    # Expand dims to broadcast and compute all pairwise distances
    mat_expanded = np.expand_dims(mat_t, 1)  # Shape: [N, 1, 2]
    all_dists = np.sqrt(np.sum((mat_expanded - mat_t)**2, axis=2))  # Shape: [N, N]

    # Identify pairs below the threshold, excluding diagonal
    ix, iy = np.triu_indices(num_points, k=1)
    valid_pairs = all_dists[ix, iy] < threshold

    # Filter pairs by distance threshold
    filtered_ix, filtered_iy = ix[valid_pairs], iy[valid_pairs]
    distances = all_dists[filtered_ix, filtered_iy]

    # Calculate direction vectors and angles
    direction_vectors = mat_t[filtered_iy] - mat_t[filtered_ix]
    angles = np.arctan2(direction_vectors[:, 1], direction_vectors[:, 0])

    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    # Normalize distances and create distance vectors
    normalized_dists = normalizeCol(distances, MIN_DIST, MAX_DIST)
    dist_vectors = np.stack([normalized_dists, cos_theta, sin_theta], axis=1)

    # Double entries for bidirectional edges
    doubled_indices = np.vstack([np.stack([filtered_ix, filtered_iy], axis=1),
                                 np.stack([filtered_iy, filtered_ix], axis=1)])
    doubled_dist_vectors = np.vstack([dist_vectors, dist_vectors * [1, -1, -1]])

    # Convert to tensors
    indices_tensor = torch.tensor(doubled_indices.T, dtype=torch.long)
    dist_tensor = torch.tensor(doubled_dist_vectors, dtype=torch.float)

    return dist_tensor, indices_tensor



def runSim(fileName, mod):
    
    sim = np.load(fileName, allow_pickle = True).item()['resOutput']
    x, y = getFeatures(sim.copy(), np.array([normalizeCol(R, MIN_RAD, MAX_RAD)]), nb = 4)
    attr, inds = [], []
    for i in range(len(x)):
        vd, vi = optimized_getGraph(sim[1+i])
        attr.append(vd)
        inds.append(vi)
        
        
    speedList = []
    gt = []
    with torch.no_grad():
        for t in range(len(x)):
            s = Data(x = x[t], edge_attr = attr[t], edge_index = inds[t], y = y[t]).to(DEVICE)
            delta = mod(s)
            speedList.append(torch.unsqueeze(delta, dim = 0))
            gt.append(torch.unsqueeze(y[t][:, :2], dim = 0))

    speedList = torch.cat(speedList, dim = 0)
    gt = torch.cat(gt, dim = 0)
        
    predsVer = sim[1:(1+speedList.shape[0]), :, :] + gt.cpu().detach().numpy()
    preds = sim[1:(1+speedList.shape[0]), :, :] +  speedList.cpu().detach().numpy()
    
    
    return preds, sim, speedList, gt


def getHeatmap(fileName, mod):
    preds, sim, speedList, gt = runSim(fileName, mod)
    
    v = sim[2:(2+speedList.shape[0]), :, :]
    
    errors = v - preds
    heatmap(errors, v, display = False)
    
    
def runSim2(fileName, mod):
    
    sim = np.load(fileName, allow_pickle = True).item()['resOutput']
    x, y = getFeatures(sim.copy(), np.array([normalizeCol(R, MIN_RAD, MAX_RAD)]), nb = 4)
    attr, inds = [], []
    for i in range(len(x)):
        vd, vi = optimized_getGraph(sim[1+i])
        attr.append(vd)
        inds.append(vi)
        
        
    speedList = []
    gt = []
    with torch.no_grad():
        for t in range(len(x)):
            s = Data(x = x[t], edge_attr = attr[t], edge_index = inds[t], y = y[t]).to(DEVICE)
            v = mod(s)
            speedList.append(torch.unsqueeze(v, dim = 0))
            gt.append(torch.unsqueeze(y[t][:, :2], dim = 0))

    speedList = torch.cat(speedList, dim = 0).to(DEVICE)
    gt = torch.cat(gt, dim = 0).to(DEVICE)
        

    
    
    return speedList, gt, sim
    
    
def getHeatmap2(fileName, mod):
    v, gt, sim = runSim2(fileName, mod)
    
    pos = sim[1:(1+v.shape[0]), :, :]
    
    errors = torch.abs(v - gt)
    heatmap(errors[8:, :, :].cpu().detach().numpy(), pos[8:, :, :], display = False)
    
    
    
    
        
    