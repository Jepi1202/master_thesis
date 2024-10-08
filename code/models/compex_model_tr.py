import torch
import torch.nn as nn
#import torch.nn.functional as F
from modules import MLP, MLP2, GN3
import yaml

import os

print('loading compex model transformer encoder version')



## import yaml cfg


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 3
MESSAGE_SHAPE = 128
HIDDEN_NN_SHAPE = 128


# output
OUTPUT_SHAPE= 2


class encoder(nn.Module):
    def __init__(self, inShape:int, latentShape:int, outputShape:int, lenSequence:int = 4, dropout = 0.5):
        super().__init__()

        self.inshape = inShape
        self.latentShape = latentShape
        self.outputShape = outputShape
        self.dropout = dropout
        self.lenSequence = lenSequence

        # update speed feature representation
        self.upscale = nn.Linear(2, latentShape)    
        tfEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=latentShape, nhead=4, dropout=dropout, batch_first=True, dim_feedforward=latentShape, activation=torch.nn.functional.leaky_relu, norm_first=False)
        self.transformerEncoder = torch.nn.TransformerEncoder(tfEncoderLayer, num_layers=6, enable_nested_tensor=True)  # need to check why 6, ...
        #self.mlp = MLP(lenSequence * latentShape, outputShape)
        self.linear2 = nn.Linear(lenSequence * latentShape, outputShape)
        self.norm1 = torch.nn.LayerNorm(outputShape)
        
    def forward(self, x):
        # x [N, T*2]
        
        y = x.view(x.shape[0], -1, 2)
        #y = self.upscale(y)     # [N, T, latentShape]
        y = self.transformerEncoder(y)    #[N, T, latenShape]
        y = y.flatten(start_dim=1)
        y = self.linear2(y)
        y = self.norm1(y)
        
        return y
        
 

class deepGNN(nn.Module):
    def __init__(self, 
                 inShape:int, 
                 latentShape:int, 
                 outShape:int, 
                 messageShape:int, 
                 edge_shape:int = EDGES_SHAPE, 
                 hiddenGN:int = 256):
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
        self.enc = encoder(inShape, (inShape + latentShape)//2, latentShape)
        
        # GNN
        self.GNNLayers = torch.nn.ModuleList()
        self.layerNormList = torch.nn.ModuleList()
        self.nbLayers = 2

        for i in range(self.nbLayers):
            self.GNNLayers.append(GN3(latentShape, messageShape, latentShape, edge_shape, hiddenGN))
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
        

        for i in range(self.nbLayers):
            y = y+self.layerNormList[i](self.GNNLayers[i](y, edge_index, edge_attr))
        
        y = self.dec(y)

        return y
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(inputShape, edge_shape = EDGES_SHAPE):
    net = deepGNN(inputShape, LATENT_SHAPE, OUTPUT_SHAPE, MESSAGE_SHAPE,edge_shape, HIDDEN_NN_SHAPE)

    return net



