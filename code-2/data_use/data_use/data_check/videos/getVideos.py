import os 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm
import re
import random

ListDatasets = ['/scratch/users/jpierre/mew_0.001_noisy_v2',
                '/scratch/users/jpierre/mew_0.001_normal_v2', 
                #'/scratch/users/jpierre/mew_0.01_noisy', 
                #'/scratch/users/jpierre/mew_0.01_normal'
               ]


outputFiles = ['/home/jpierre/v2/data_check/videos/mew_0.001_noisy_v2',
                '/home/jpierre/v2/data_check/videos/mew_0.001_normal_v2', 
                #'/home/jpierre/v2/data_check/videos/mew_0.01_noisy', 
                #'/home/jpierre/v2/data_check/videos/mew_0.01_normal'
              ]


SUB_CATEGORY = ['training', 'validation', 'test']

CFG_VIDEO = {'radius': 5, 'color': (0, 255, 0)}


class videoParameters():
    def __init__(self, path:str, params = CFG_VIDEO):
        """ 
        Args:
        -----
            - `path`: path of the video
        """
        self.fps = 10
        self.path = path
        self.size = (600, 600)
        self.params = params


def display_out_graph(data_graph, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing graph nodes and edges using cv2,
    based on specified video parameters and position bounds.

    Parameters:
    - data_x: List of input features of the graph
    - edge_indices: List of tensors for the different indices
    - video_params: An instance of VideoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, uses the minimum and maximum values from the data.
    """


    ## get back the array with teh positions

    mat_pos = []
    mat_indices = []

    for i in range(len(data_graph)):
        mat_pos.append(data_graph[i].x[:, :2])

        mat_indices.append(data_graph[i].edge_index)    


    mat_pos = np.stack(mat_pos, axis = 0)  
    #print(len(mat_indices))
    #print(mat_pos.shape)
        
    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = mat_pos[:, :, 0].min(), mat_pos[:, :, 0].max()
        min_y, max_y = mat_pos[:, :, 1].min(), mat_pos[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    mat_pos[:, :, 0] = (mat_pos[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    mat_pos[:, :, 1] = (mat_pos[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)
    
    mat_pos = mat_pos.astype(int)

    for i in range(mat_pos.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)

        # Draw nodes
        for x, y in mat_pos[i]:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw edges
        for start_node, end_node in mat_indices[i].t().numpy():
            pt1 = tuple(mat_pos[i, start_node, :])
            pt2 = tuple(mat_pos[i, end_node, :])
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)


        out.write(frame)

    out.release()
    
    
def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
        
def get_number_sim(path):
    pattern = re.compile(f"sim_(\d+)\_step_0.pt")
    
    sim_nb_list = []
    
    for filename in os.listdir(path):
        match = pattern.match(filename)
        if match:
            nb_sim = int(match.group(1))
            sim_nb_list.append(nb_sim)
            
    sim_nb_list.sort()
    
    return sim_nb_list
            
        
def collect_steps(path, simulation_number):
    pattern = re.compile(f"sim_{simulation_number}_step_(\d+)\.pt")

    steps = []; res = []

    # Iterate over all files in the directory
    for filename in os.listdir(path):
        # Check if the file matches the pattern
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            steps.append(step)
    
    steps.sort()
    
    for i in range(len(steps)):
        res.append(torch.load(os.path.join(path, f"sim_{simulation_number}_step_{steps[i]}.pt")))
    
    return res
        

def getVideos(path, outFile, limit = 3):
    
    makeDir(outFile)
    
    for cat in SUB_CATEGORY:
        p = os.path.join(outFile, cat)
               
        makeDir(p)
        
        
        p_torch = os.path.join(path, cat)
        p_torch = os.path.join(p_torch, 'torch_file')
        
  
        # get the number of simulations
        
        simulation_numbers = get_number_sim(p_torch)
        
        random.shuffle(simulation_numbers)
        
        simulation_numbers = simulation_numbers[:limit]
        
        for i in tqdm(range(len(simulation_numbers))):
            nb_s = simulation_numbers[i]
            
            # collect the steps and associtated data
            
            r = collect_steps(p_torch, nb_s)
            
            params = videoParameters(path = os.path.join(p, f'simulation_{nb_s}.mp4'))
            
            display_out_graph(r, params)

       

        
def main():
    for i in range(len(ListDatasets)):
        datas = ListDatasets[i]
        output = outputFiles[i]
        
        getVideos(datas, output)
        


if __name__ == '__main__':
    main()
    