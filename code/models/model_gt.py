import torch
import torch.nn as nn
from modules import *
import yaml
import os



class model_gt(nn.Module):
    def __init__(self):
        """ 
        Neural network to combine everything

        """
        
        super().__init__()
        
    def forward(self, nextSpped):
        """ 
        
        Args:
        -----
        - `x`: value for the nodes [# Nodes, #Timesteps x inShape]
        - `edge_index`
        - `edge_attr`
        """
       
        return nextSpped
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(inputShape = None, edge_shape = None):
    net = model_gt()

    return net