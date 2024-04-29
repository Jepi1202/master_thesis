import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils import degree as deg
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GeneralConv
from torch_geometric.data import Dataset, Data
import torch.utils.data as torchData
from torch.utils.data import Dataset as torchDataset
from torch_geometric.loader import DataLoader


from tqdm import tqdm
import os
import wandb


class MLP(nn.Module):
    """ 
    linearly growing size
    """
    def __init__(self, inputShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.inputShape = inputShape
        self.outputShape = outputShape

        self.delta = (inputShape - outputShape) // 3
        dim1 = inputShape - self.delta
        dim2 = dim1 - self.delta

        self.mlp = nn.Sequential(
            nn.Linear(inputShape, dim1),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim1, dim2),
            #nn.ELU(),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim2, outputShape),
        )
        
        self.init_weights()
    
    def forward(self, x):
        x = self.mlp(x)
        return x
    
    
    def init_weights(self):
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                #nn.init.xavier_normal_(layer.weight)
                #nn.init.zeros_(layer.bias)
                layer.bias.data.fill_(0.)



class MLP2(nn.Module):
    """
    constant size
    """
    def __init__(self, inputShape:int, latentShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP2, self).__init__()

        self.dropout = dropout
        self.inputShape = inputShape
        self.latentShape = latentShape
        self.outputShape = outputShape

        self.mlp = nn.Sequential(
            nn.Linear(inputShape, latentShape),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latentShape, latentShape),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latentShape, outputShape),
        )
        
        self.init_weights()
    
    def forward(self, x):
        x = self.mlp(x)
        return x
    
    def init_weights(self):
        for layer in self.mlp.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                #nn.init.xavier_normal_(layer.weight)
                #nn.init.zeros_(layer.bias)
                layer.bias.data.fill_(0.)




class GN(MessagePassing):
    def __init__(self, inputShape:int, messageShape:int, outputShape:int, shapeEdges:int = 7, hiddenShape:int=256, aggr:str='add'):
        super(GN, self).__init__(aggr=aggr)

        self.inputShape = inputShape
        self.messageShape = messageShape
        self.outputShape = outputShape
        self.hiddenShape = hiddenShape
        
        self.messageMLP = MLP2(2*inputShape + shapeEdges, hiddenShape, messageShape)
        self.norm = torch.nn.LayerNorm(messageShape)
        
        self.updateMLP = MLP(messageShape + inputShape, outputShape)
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        """ 
        Forward of the Graph neural netowk
        Relies on .forward that will use message, aggregate and update
        
        Args:
        ----- 
        - `x`: tensor of nodes [# Nodes, inputShape]
        - `edge_index`: tensor of the edges of the graph
        
        Returns:
        --------
        Tensor of shape [# Nodes, outputShape] computed according to the GN formula
        """
        # nb: no automatic self loop here ...
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)),edge_attr=edge_attr, x=x)
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """
        

        xVal = torch.cat([x_i, x_j], dim=-1)  #  [# edges to i, 2*inputShape]
        xVal = torch.cat([xVal, edge_attr], dim=-1)
        y = self.norm(self.messageMLP(xVal))
        return y
    
    def update(self, aggr_out:torch.tensor, x:torch.tensor):
        """ 
        Function to update all the nodes after the aggregation
        
        Args:
        -----
        - `aggr_out`: result after the aggregation [# Nodes, messageShape]
        - `x`: current node [1, inputShape]
        """

        xVal = torch.cat([x, aggr_out], dim=-1)
        return self.updateMLP(xVal) 
    
    
    
class GN2(MessagePassing):
    def __init__(self, inputShape:int, messageShape:int, outputShape:int, shapeEdges:int = 7, hiddenShape:int=256, aggr:str='add'):
        super(GN2, self).__init__(aggr=aggr)

        self.inputShape = inputShape
        self.messageShape = messageShape
        self.outputShape = outputShape
        self.hiddenShape = hiddenShape
        
        self.messageMLP = MLP2(2*inputShape + shapeEdges, hiddenShape, messageShape)
        self.norm = torch.nn.LayerNorm(messageShape)
        
        self.updateMLP = MLP(messageShape + inputShape, outputShape)
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        """ 
        Forward of the Graph neural netowk
        Relies on .forward that will use message, aggregate and update
        
        Args:
        ----- 
        - `x`: tensor of nodes [# Nodes, inputShape]
        - `edge_index`: tensor of the edges of the graph
        
        Returns:
        --------
        Tensor of shape [# Nodes, outputShape] computed according to the GN formula
        """
        # nb: no automatic self loop here ...
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)),edge_attr=edge_attr, x=x)
    
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        # Identify isolated nodes
        degree = deg(edge_index[0], num_nodes=x.size(0))
        isolated_nodes_mask = degree == 0
        
        # Propagate messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, x=x)
        
        # Set aggregated output for isolated nodes to null vector
        if isolated_nodes_mask.any():
            null_vector = torch.zeros_like(out[isolated_nodes_mask])
            out[isolated_nodes_mask] = null_vector
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """
        

        xVal = torch.cat([x_i, x_j], dim=-1)  #  [# edges to i, 2*inputShape]
        xVal = torch.cat([xVal, edge_attr], dim=-1)
        y = self.norm(self.messageMLP(xVal))
        return y
    
    def update(self, aggr_out:torch.tensor, x:torch.tensor):
        """ 
        Function to update all the nodes after the aggregation
        
        Args:
        -----
        - `aggr_out`: result after the aggregation [# Nodes, messageShape]
        - `x`: current node [1, inputShape]
        """

        xVal = torch.cat([x, aggr_out], dim=-1)
        return self.updateMLP(xVal) 
    
    
    

    
class GN3(MessagePassing):
    def __init__(self, inputShape:int, messageShape:int, outputShape:int, shapeEdges:int = 7, hiddenShape:int=64, aggr:str='add'):
        super(GN3, self).__init__(aggr=aggr)

        self.inputShape = inputShape
        self.messageShape = messageShape
        self.outputShape = outputShape
        self.hiddenShape = hiddenShape
        
        self.messageMLP = MLP2(shapeEdges, hiddenShape, messageShape)
        self.norm = torch.nn.LayerNorm(messageShape)
        
        self.updateMLP = MLP(messageShape + inputShape, outputShape)
    
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        # Identify isolated nodes
        degree = deg(edge_index[0], num_nodes=x.size(0))
        isolated_nodes_mask = degree == 0
        
        # Propagate messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, x=x)
        
        # Set aggregated output for isolated nodes to null vector
        if isolated_nodes_mask.any():
            null_vector = torch.zeros_like(out[isolated_nodes_mask])
            out[isolated_nodes_mask] = null_vector
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """
        
        y = self.norm(self.messageMLP(edge_attr))
        return y
    
    def update(self, aggr_out:torch.tensor, x:torch.tensor):
        """ 
        Function to update all the nodes after the aggregation
        
        Args:
        -----
        - `aggr_out`: result after the aggregation [# Nodes, messageShape]
        - `x`: current node [1, inputShape]
        """

        xVal = torch.cat([x, aggr_out], dim=-1)
        return self.updateMLP(xVal) 