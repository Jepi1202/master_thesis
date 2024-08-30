import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GATv2Conv, GeneralConv
from torch_geometric.data import Dataset, Data
import torch.utils.data as torchData
from torch.utils.data import Dataset as torchDataset
from torch_geometric.loader import DataLoader


from tqdm import tqdm
import os
import wandb


from modules import *


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 3
MESSAGE_SHAPE = 64
HIDDEN_NN_SHAPE = 64


# output
OUTPUT_SHAPE= 2


       
 

class GN_NN(nn.Module):
    def __init__(self, inShape:int, latentShape:int, outShape:int, messageShape:int, edge_shape:int = EDGES_SHAPE, hiddenGN:int = 128):
        """ 
        Neural network to combine everything
        
        Args:
        -----
        - `inShape`: shape of the input vector
        - `latentShape`: shape of the latent space
        - `outShape`: shape of the output vector
        - `messageShape`: shape of the message in the GN
        - `hiddenGN`: shape of the hidden layers in the MLP of the GN
        """
        
        super().__init__()
        
        self.inShape = inShape
        self.latentShape = latentShape
        self.outShape = outShape
        
        ## encoder
        self.enc = MLP(inShape, latentShape)
        self.norm1 = torch.nn.LayerNorm(latentShape)
        
        # GNN
        self.GNN = GN2(latentShape, messageShape, outShape, edge_shape, hiddenGN)
        
    def forward(self, graph):
        """ 
        
        Args:
        -----
        - `x`: value for the nodes [# Nodes, #Timesteps x inShape]
        - `edge_index`
        - `edge_attr`
        """

        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        
        
        v_x = torch.mean(torch.tensor([x[:, 0], x[:, 2], x[:, 4], x[:, 6]]))
        v_y = torch.mean(torch.tensor([x[:, 1], x[:, 3], x[:, 5], x[:, 7]]))
        v = torch.tensor([v_x, v_y])

        return v
    

    def L1Reg(self, graph):
        nodes1 = graph.x[graph.edge_index[0, :]]
        nodes2 = graph.x[graph.edge_index[1, :]]
        atr = graph.edge_attr
        
        self.enc.eval()
        with torch.no_grad():
            nodes1 = self.norm1(self.enc(nodes1))
            nodes2 = self.norm1(self.enc(nodes2))
        self.enc.train()

        messages = self.GNN.message(nodes1, nodes2, atr)

        loss = 0.01 * torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        return loss


def loadNetwork(inputShape, edge_shape = EDGES_SHAPE):
    net = GN_NN(inputShape, LATENT_SHAPE, OUTPUT_SHAPE, MESSAGE_SHAPE,edge_shape, HIDDEN_NN_SHAPE)

    return net