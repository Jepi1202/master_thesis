import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing, GATv2Conv

import os


from modules import *


# load the cfg




LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 5
MESSAGE_SHAPE = 128
HIDDEN_NN_SHAPE = 128

NB_GNN = 3


# output
OUTPUT_SHAPE= 2


class encoder(nn.Module):
    def __init__(self, inShape:int, latentShape:int):
        super().__init__()
        self.mlp = MLP(inShape, latentShape)
        self.norm1 = torch.nn.LayerNorm(latentShape)
        
    def forward(self, x):
        y = self.mlp(x)                                                    # [#Nodes, latentShape]
        y = self.norm1(y)
        
        return y
        
 

class GN_NN(nn.Module):
    def __init__(self, inShape:int, latentShape:int, outShape:int, messageShape:int, edge_shape:int = EDGES_SHAPE, hiddenGN:int = 256, nb_gnn = NB_GNN):
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
        self.enc = encoder(inShape, latentShape)
        
        # GNN
        self.GNNLayers = torch.nn.ModuleList()
        self.layerNormList = torch.nn.ModuleList()
        self.nb_gnn = nb_gnn
        
        for i in range(nb_gnn):
            #self.GNNLayers.append(GN(latentShape, messageShape, latentShape, edge_shape, hiddenGN))
            self.GNNLayers.append(GATv2Conv(latentShape, latentShape, heads=1, dropout=0.4, concat=True, edge_dim=edge_shape, add_self_loops=False, fill_value=0.0))
            self.layerNormList.append(torch.nn.LayerNorm(latentShape))
    
        
        self.dec = MLP(latentShape, outShape)
        
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
        
        
        # encoder part
        y = self.enc(x)                                                    # [#Nodes, latentShape]
        
        # gnn part
        for i in range(self.nb_gnn):
            y = y+self.layerNormList[i](self.GNNLayers[i](y, edge_index, edge_attr))
        
        y = self.dec(y)

        return y
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(inputShape, edge_shape = EDGES_SHAPE):
    print(">>>>>>>>>>> Loading GaT model")
    net = GN_NN(inputShape, LATENT_SHAPE, OUTPUT_SHAPE, MESSAGE_SHAPE,edge_shape, HIDDEN_NN_SHAPE)

    return net