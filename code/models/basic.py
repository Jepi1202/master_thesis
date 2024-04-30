import torch
import torch.nn as nn
from modules import *
import yaml
import os



class basic(nn.Module):
    def __init__(self):
        """ 
        Neural network to combine everything

        """
        
        super().__init__()
        
    def forward(self, graph):
        """ 
        
        Args:
        -----
        - `x`: value for the nodes [# Nodes, #Timesteps x inShape]
        - `edge_index`
        - `edge_attr`
        """

        x = graph.x
       
        return x[:, :2]
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(inputShape = None, edge_shape = None):
    net = basic()

    return net