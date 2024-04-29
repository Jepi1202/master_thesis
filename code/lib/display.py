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
        