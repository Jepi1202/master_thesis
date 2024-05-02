import matplotlib.pyplot as plt
import cv2
import numpy as np


# add bounds ...
CFG_VIDEO = {'radius': 5, 'color': (0, 255, 0)}


def linePlot(x:np.array, 
             y:np.array,
             outputPath:str = 'linePlot.png',
             display:bool = True,
             xlabel:str = 'x',
             ylabel:str = 'y'
             ):
    

    plt.plot(x, y, zorder = 2, xlabel = xlabel, ylabel = ylabel)
    plt.grid(zorder = 1)
    
    if display:
        plt.show()

    else:
        plt.savefig(outputPath)
        plt.close()



def create_simulation_video_cv2(data, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements using cv2,
    based on specified video parameters and position bounds.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - video_params: An instance of videoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, it uses the minimum and maximum values from the data.
    """
    
    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = data[:, :, 0].min(), data[:, :, 0].max()
        min_y, max_y = data[:, :, 1].min(), data[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    data[:, :, 0] = (data[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    data[:, :, 1] = (data[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)

    for i in range(data.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)
        for x, y in data[i]:
            # Draw the cell as a circle
            cv2.circle(frame, (int(x), int(y)), radius=video_params.params['radius'], color=video_params.params['color'], thickness=-1)
        out.write(frame)

    out.release()



# test needed

def videoGraphBase(data, edge_indices, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing graph nodes and edges using cv2,
    based on specified video parameters and position bounds.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of nodes, and 2 corresponds to the coordinates (x, y).
    - edge_indices: A PyTorch tensor of shape [2, E], where E is the number of edges,
                    and each column is a pair of indices (start_node, end_node).
    - video_params: An instance of VideoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, uses the minimum and maximum values from the data.
    """
    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = data[:, :, 0].min(), data[:, :, 0].max()
        min_y, max_y = data[:, :, 1].min(), data[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    data[:, :, 0] = (data[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    data[:, :, 1] = (data[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)

    for i in range(data.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)

        # Draw nodes
        for x, y in data[i].int().numpy():
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw edges
        for start_node, end_node in edge_indices[i].t().numpy():
            pt1 = tuple(data[i, start_node].int().numpy())
            pt2 = tuple(data[i, end_node].int().numpy())
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)


        out.write(frame)

    out.release()



## test needed
from matplotlib.colors import Normalize

def videoGraphColor(data, edge_indices, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing graph nodes and edges using cv2,
    based on specified video parameters and position bounds, with edge colors indicating distances.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of nodes, and 2 corresponds to the coordinates (x, y).
    - edge_indices: A PyTorch tensor of shape [2, E], where E is the number of edges,
                    and each column is a pair of indices (start_node, end_node).
    - video_params: An instance of VideoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, uses the minimum and maximum values from the data.
    """
    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = data[:, :, 0].min(), data[:, :, 0].max()
        min_y, max_y = data[:, :, 1].min(), data[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    data[:, :, 0] = (data[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    data[:, :, 1] = (data[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Calculate distances and set up colormap
    max_dist = 10       # 6 actually
    #max_dist = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    norm = Normalize(vmin=0, vmax=max_dist)
    cmap = plt.get_cmap('jet')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)

    for i in range(data.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)

        # Draw edges
        for start_node, end_node in edge_indices.t().numpy():
            pt1 = tuple(data[i, start_node].int().numpy())
            pt2 = tuple(data[i, end_node].int().numpy())
            dist = np.linalg.norm(data[i, start_node] - data[i, end_node])
            color = cmap(norm(dist))[:3]  # Get RGB from RGBA
            color = tuple(int(255 * c) for c in color[::-1])  # Convert to BGR for OpenCV
            cv2.line(frame, pt1, pt2, color, 1)

        # Draw nodes
        for x, y in data[i].int().numpy():
            cv2.circle(frame, (x, y), video_params.radius, video_params.color, -1)

        out.write(frame)

    out.release()


# need test

def videoColors(data, errors, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements using cv2,
    with cell colors indicating error values, based on specified video parameters and position bounds.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - errors: A PyTorch tensor of shape [T, N], representing the error of each cell at each timestep.
    - video_params: An instance of videoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, it uses the minimum and maximum values from the data.
    """

    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = data[:, :, 0].min(), data[:, :, 0].max()
        min_y, max_y = data[:, :, 1].min(), data[:, :, 1].max()

    # Normalize coordinates to fit within the video frame size
    data[:, :, 0] = (data[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    data[:, :, 1] = (data[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Normalize errors for color mapping
    norm = Normalize(vmin=errors.min(), vmax=errors.max())
    cmap = plt.get_cmap('hot')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)

    for i in range(data.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)
        for idx, (x, y) in enumerate(data[i]):
            color_value = cmap(norm(errors[i, idx]))[:3]  # Get RGB from RGBA
            color = tuple(int(255 * c) for c in color_value[::-1])  # Convert to BGR for OpenCV
            cv2.circle(frame, (int(x), int(y)), radius=video_params.params['radius'], color=color, thickness=-1)
        out.write(frame)

    out.release()


# need testing


def compareVideo(data, ground_truth, video_params, bounds=None):
    """
    Creates an MP4 video from a PyTorch tensor representing cell movements using cv2,
    and includes ground truth data visualized in a different color.

    Parameters:
    - data: A PyTorch tensor of shape [T, N, 2], where T is the number of timesteps,
            N is the number of cells, and 2 corresponds to the coordinates (x, y).
    - ground_truth: A PyTorch tensor of shape [T, N, 2], same format as data, representing the ground truth.
    - video_params: An instance of videoParameters class containing video settings.
    - bounds: Tuple of ((min_x, max_x), (min_y, max_y)) specifying the bounds for the positions.
              If None, it uses the minimum and maximum values from the data.
    """
    
    if bounds:
        min_x, max_x = bounds[0]
        min_y, max_y = bounds[1]
    else:
        min_x, max_x = min(data.min(), ground_truth.min()), max(data.max(), ground_truth.max())
        min_y, max_y = min(data.min(), ground_truth.min()), max(data.max(), ground_truth.max())

    # Normalize coordinates to fit within the video frame size
    data[:, :, 0] = (data[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    data[:, :, 1] = (data[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)
    ground_truth[:, :, 0] = (ground_truth[:, :, 0] - min_x) / (max_x - min_x) * (video_params.size[0] - 1)
    ground_truth[:, :, 1] = (ground_truth[:, :, 1] - min_y) / (max_y - min_y) * (video_params.size[1] - 1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lowercase
    out = cv2.VideoWriter(video_params.path, fourcc, video_params.fps, video_params.size)

    # Colors (BGR format) and radius
    pred_color = (0, 255, 0)  # Green for predictions
    gt_color = (0, 0, 255)  # Red for ground truth

    for i in range(data.shape[0]):
        frame = np.zeros((video_params.size[1], video_params.size[0], 3), dtype=np.uint8)
        
        # Draw ground truth and predictions
        for (x, y), (gt_x, gt_y) in zip(data[i], ground_truth[i]):
            cv2.circle(frame, (int(x), int(y)), radius=5, color=pred_color, thickness=-1)
            cv2.circle(frame, (int(gt_x), int(gt_y)), radius=2, color=gt_color, thickness=-1)
        
        # Draw legend
        cv2.rectangle(frame, (10, 10), (10 + 20, 30), pred_color, -1)  # Prediction color box
        cv2.rectangle(frame, (10, 40), (10 + 20, 60), gt_color, -1)  # Ground truth color box
        cv2.putText(frame, 'Prediction', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Ground Truth', (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()


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


class Displayer():
    def __init__(self):

        self.videoFun = None
        self.videoParams = None

        self.graphFun = None
        self.graphParams = None


    def addVideo(self, fun,  params):
        if not self.videoFun:
            self.videoFun = []
            self.videoParams = []

        self.videoFun.append(fun)
        self.videoParams.append(params)


    def addGraph(self, fun,  params):
        if not self.graphFun:
            self.graphFun = []
            self.graphParams = []

        self.graphFun.append(fun)
        self.graphParams.append(params)



    def display(self):

        if self.videoFun is not None:
            for i in range(len(self.videoFun)):
                self.videoFun[i](*self.videoParams[i])


        if self.graphFun is not None:
            for i in range(len(self.graphFun)):
                self.graphFun[i](*self.graphParams[i])
        