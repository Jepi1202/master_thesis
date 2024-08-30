import matplotlib.pyplot as plt
#import networkx as nx
import matplotlib.colors as mcolors
import cv2
import numpy as np


"""
def display_graph(edges, node_positions, edge_values, minVal = 0, maxVal = 2, display = False, colorbar = False):
    G = nx.Graph()
    G.add_nodes_from(node_positions.keys())
    G.add_edges_from(edges)

    # Choose a colormap, for example, 'viridis'
    cmap = plt.cm.jet

    # Normalize values to range [0, 1] for colormap
    norm = mcolors.Normalize(vmin=minVal, vmax=maxVal)

    # Create a colormap scalar mappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Normalize edge values manually
    normalized_edge_values = [norm(val) for val in edge_values]

    # Draw the graph with colored edges
    nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=50, node_color='skyblue', font_color='black', font_size=8, edgelist=edges, edge_color=normalized_edge_values, cmap=cmap, linewidths=2)
    # nx.draw(G,  with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8, edgelist=edges, edge_color=normalized_edge_values, cmap=cmap, linewidths=2)

    # Add a colorbar to show the mapping of values to colors
    if colorbar:
        plt.colorbar(sm, label='Edge Values')
    
    if display:
        plt.show()
"""


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

def create_simulation_video_cv2_norm(data, filename='simulation.mp4', fps=10, size=(600, 600), bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements using cv2.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - filename: Name of the output MP4 file.
    - fps: Frames per second for the output video.
    - size: Size of the output video frame.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, it uses the minimum and maximum values from the data.
    """
    # Convert the data to numpy for easier manipulation
    data_np = data.numpy()

    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = data_np[:, :, 0].min(), data_np[:, :, 0].max()
        min_y, max_y = data_np[:, :, 1].min(), data_np[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    data_np[:, :, 0] = (data_np[:, :, 0] - min_x) / (max_x - min_x) * (size[0] - 1)
    data_np[:, :, 1] = (data_np[:, :, 1] - min_y) / (max_y - min_y) * (size[1] - 1)

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

# Example usage:
# data = torch.tensor(...)  # Some PyTorch tensor with shape [T, N, 2]
# create_simulation_video_cv2(data, bounds=((0, 100), (0, 100)))
