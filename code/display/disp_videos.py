import cv2
import numpy as np
import torch
import argparse
import os
from tqdm import tqdm

THRESHOLD_DIST = 6



# python disp_video.py --path '/scratch/users/jpierre/test_speed_easy/test/np_file//output_1.npy'

def retrieveArgs():
    """ 
    Function to retrieve the args sent to the python code
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True, default= None,
                        help="Path of the video to create")
   
    parser.add_argument("--output", type=str, default= 'sim.mp4',
                        help="output path of the video")
    
    parser.add_argument("--outputGraph", type=str, default= 'simGraph.mp4',
                        help="output path of the video")
    
    parser.add_argument("--fps", type=str, default= 10,
                        help="fps of the video")

    args = vars(parser.parse_args())

    return args



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
    
    
    
import cv2
import numpy as np
import torch

def create_simulation_video_cv2_with_edge_colors(data, edge_indices_list, edge_arrays_list, filename='simulation_with_edge_colors.mp4', fps=10, size=(600, 600)):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements and their edges using cv2,
    with edge colors determined by the first element of edge arrays.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - edge_indices_list: A list of length T, each a numpy array of shape [2, #edges] indicating the indices of the edges for each timestep.
    - edge_arrays_list: A list of length T, each a numpy array of shape [#edges, K] with K the dimensions of the array for each edge, for each timestep.
    - filename: Name of the output MP4 file.
    - fps: Frames per second for the output video.
    - size: Size of the output video frame.
    """
    data_np = data.numpy() if isinstance(data, torch.Tensor) else data

    # Normalize coordinates to fit within the video frame size
    data_np -= data_np.min(axis=(0, 1), keepdims=True)
    data_np /= data_np.max(axis=(0, 1), keepdims=True)
    data_np *= np.array([size[0] - 1, size[1] - 1])
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(filename, fourcc, fps, size)
    
    # Extract all first elements to find global min and max
    minArray = np.array([np.min(arr[:, 0].numpy()) for arr in edge_arrays_list])
    maxArray = np.array([np.min(arr[:, 0].numpy()) for arr in edge_arrays_list])
    min_val, max_val = np.min(minArray), np.max(maxArray)
    print(min_val, max_val)
    
    for t in tqdm(range(data_np.shape[0])):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        edge_indices = edge_indices_list[t].numpy()
        edge_arrays = edge_arrays_list[t].numpy()
        
        # Draw edges based on the first element of edge arrays
        for edge_idx, (i, j) in enumerate(edge_indices.T):
            x1, y1 = data_np[t, i]
            x2, y2 = data_np[t, j]
            # Normalize the first element to [0, 1] for color mapping
            val_normalized = (edge_arrays[edge_idx][0] - min_val) / (max_val - min_val)
            # Map the normalized value to a color gradient (blue to red here)
            color = (255 * (1 - val_normalized), 0, 255 * val_normalized)
            #print(color)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=3)
        
        # Draw cells
        for x, y in data_np[t]:
            cv2.circle(frame, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
        
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

    normalized_dists = distances

    dist_vectors = np.stack([normalized_dists, cos_theta, sin_theta], axis=1)

    # Double entries for bidirectional edges
    doubled_indices = np.vstack([np.stack([filtered_ix, filtered_iy], axis=1),
                                 np.stack([filtered_iy, filtered_ix], axis=1)])
    doubled_dist_vectors = np.vstack([dist_vectors, dist_vectors * [1, -1, -1]])

    # Convert to tensors
    indices_tensor = torch.tensor(doubled_indices.T, dtype=torch.long)
    dist_tensor = torch.tensor(doubled_dist_vectors, dtype=torch.float)

    return dist_tensor, indices_tensor
    
    
def main():
    args = retrieveArgs()
    sim = args['path']
    outputPath = args['output']
    outputGraph = args['outputGraph']
    fps = args['fps']
    
    # load the simulation
    assert os.path.exists(sim)
    sim = np.load(sim, allow_pickle=True)
    sim = sim.item()['resOutput']
    
    create_simulation_video_cv2(sim.copy(), filename = outputPath, fps = fps)
    
    edge_index = []
    edge_attr = []
    for i in range(sim.shape[0]):
        edg, ind = optimized_getGraph(sim[i])
        edge_index.append(ind) 
        edge_attr.append(edg)
                   
    create_simulation_video_cv2_with_edge_colors(sim.copy(), edge_index, edge_attr, outputGraph)
    
if __name__ == '__main__':
    main()