import torch
import numpy as np
import matplotlib.pyplot as plt

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
        
        mean_acceleration_grid[count_grid == 0] = np.nan  # Set cells with no data to NaN

        # Plotting the heatmap
        plt.figure(figsize=plot_size)
        plt.imshow(mean_acceleration_grid.T, origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', cmap='jet')
        plt.colorbar(label='Error')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Heatmap of Error')
    
        plt.show()





def MSE_rollout(roll, sim):
    x = np.arange(roll.shape[0])
    vals = ((roll - sim) ** 2).reshape(x.shape[0], -1)
    y = np.mean(vals, axis=1)
    std = np.std(vals, axis=1)

    plt.plot(x, y, 'blue')
    plt.fill_between(x, y-std, y+std)
    plt.xlabel('Time')
    plt.ylabel('Rollout MSE')

    return x, y