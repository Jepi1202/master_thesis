import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import yaml

import os

print('loading compex model')



## import yaml cfg


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 3
MESSAGE_SHAPE = 128
HIDDEN_NN_SHAPE = 128


# output
OUTPUT_SHAPE= 2

GNN_CFG = {
    "input_shape": 8,
    "edges_shape": 5,
    "output_shape": 2,
    "nb_layers":3,
    "layer_norm":1,
    "encoder":{
        "presence": "no",
        "out_shape": 128,
        "dropout": "no",
        "latent_edge": 128,
    },
    "layer": {
            "input_shape": 128,
            "edges_shape": 128,
            "message_shape": 128,
            "hidden_shape":128,
            "dropout": 'no',
            "MLP_message": {
                "hidden_shape": 128,
                "message_shape": 128,
                "dropout": "no",
            },
            "MLP_update": {
                "hidden_shape": 128,
                "dropout": "no",
            },
            "regularization": {
                "name": "l1",
                "scaler": 0.01,
            }
    },
    "decoder":{
        "presence": "no",
        "hidden_shape": 128,
        "dropout": "no",
    },
    "regularization": {
        "name": "l1",
        "scaler": 0.01,
    }
}


class MLP(nn.Module):
    """ 
    linearly growing size
    """
    def __init__(self, inputShape:int, outputShape:int, dropout:float = 0.3, debug = False):
        super(MLP, self).__init__()

        self.dropout = dropout
        self.inputShape = inputShape
        self.outputShape = outputShape

        self.delta = (inputShape - outputShape) // 3
        dim1 = inputShape - self.delta
        dim2 = dim1 - self.delta


        if debug:
            print(dropout)
        
        
        mods = []
        mods.append(nn.Linear(inputShape, dim1))
        mods.append(nn.LeakyReLU())

        if not isinstance(dropout, str):
            mods.append(nn.Dropout(p=dropout))
        
        mods.append(nn.Linear(dim1, dim2)),
        #nn.ELU(),
        mods.append(nn.LeakyReLU()),
        if not isinstance(dropout, str):

            mods.append(nn.Dropout(p=dropout))

        mods.append(nn.Linear(dim2, outputShape))

        self.mlp = nn.Sequential(*mods)
        
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
                layer.bias.data.fill_(0.01)
 


class MLP2(nn.Module):
    """
    constant size
    """
    def __init__(self, inputShape:int, latentShape:int, outputShape:int, dropout:float = 0.3, debug = False):
        super(MLP2, self).__init__()

        self.dropout = dropout
        self.inputShape = inputShape
        self.latentShape = latentShape
        self.outputShape = outputShape

        if debug:
            print(dropout)

        mods = []
        mods.append(nn.Linear(inputShape, latentShape))
        mods.append(nn.LeakyReLU())
        
        if not isinstance(dropout, str):

            mods.append(nn.Dropout(p=dropout))

        mods.append(nn.Linear(latentShape, latentShape))
        mods.append(nn.LeakyReLU())
        
        if not isinstance(dropout, str):
            mods.append(nn.Dropout(p=dropout))

        mods.append(nn.Linear(latentShape, outputShape))

        self.mlp = nn.Sequential(*mods)
        
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
                layer.bias.data.fill_(0.01)

class GN_edge(MessagePassing):
    """ 
    Message passing neural network in which the message passing
    only considers the edges features
    """
    def __init__(self, d, aggr:str='add'):
        super(GN_edge, self).__init__(aggr=aggr)

        self.inputShape = d['input_shape']
        self.edgeShape = d['edges_shape']
        self.outputShape = d['output_shape']

        self.messageShape = d['MLP_message']['message_shape']
        self.hiddenShape = d['MLP_message']['hidden_shape']
        
        self.messageMLP = MLP2(inputShape = self.edgeShape, 
                               latentShape = self.hiddenShape, 
                               outputShape = self.messageShape, 
                               dropout = d['MLP_message']['dropout'])
        
        self.norm = torch.nn.LayerNorm(self.messageShape + self.inputShape)
        
        self.updateMLP = MLP(self.messageShape + self.inputShape, self.outputShape)
    
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        # Propagate messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, x=x)
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """
        
        y = self.messageMLP(edge_attr)
        return y
    
    def update(self, aggr_out:torch.tensor, x:torch.tensor):
        """ 
        Function to update all the nodes after the aggregation
        
        Args:
        -----
        - `aggr_out`: result after the aggregation [# Nodes, messageShape]
        - `x`: current node [1, inputShape]
        """

        xVal = self.norm(torch.cat([x, aggr_out], dim=-1))
        return self.updateMLP(xVal) 

                           
                

class encoder(nn.Module):
    def __init__(self, inShape:int, latentShape:int):
        super().__init__()
        self.mlp = MLP(inShape, latentShape)
        self.norm1 = torch.nn.LayerNorm(latentShape)
        
    def forward(self, x):
        y = self.mlp(x)                                                    # [#Nodes, latentShape]
        y = self.norm1(y)
        
        return y
        
 

class deepGNN(nn.Module):
    def __init__(self, 
                 d):
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
        
        self.inShape = d['input_shape']
        self.edge_shape = d['edges_shape']
        self.outShape = d['output_shape']
                
        ## encoder
        if d['encoder']['presence']:
            self.enc = nn.Sequential(MLP(inputShape = self.inShape, 
                                    outputShape = d['encoder']['out_shape'],
                                    dropout = d['encoder']['dropout']),
                                    nn.LayerNorm(d['encoder']['out_shape']))

            self.enc_edge = nn.Sequential(MLP(inputShape = self.edge_shape, 
                                    outputShape = d['encoder']['out_shape'],
                                    dropout = d['encoder']['dropout']),
                                    nn.LayerNorm(d['encoder']['out_shape']))    
        else:
            self.enc = nn.Identity()
            self.enc_edge = nn.Identity()


        d['layer']['output_shape'] = 128
        d['layer']['edge_shape'] = d['encoder']['out_shape']
        
        # GNN
        self.GNNLayers = torch.nn.ModuleList()
        self.layerNormList = torch.nn.ModuleList()
        self.nbLayers = d['nb_layers']

        for i in range(self.nbLayers):
            self.GNNLayers.append(GN_edge(d['layer']))
            if d['layer_norm']:
                self.layerNormList.append(torch.nn.LayerNorm(d['layer']['output_shape']))
            else:
                self.layerNormList.append(nn.Identity())
        
        
        if d['encoder']['presence']:
            self.dec =MLP(inputShape = d['layer']['output_shape'], 
                        outputShape = self.outShape,
                        dropout = d['decoder']['dropout'])

        else:
            self.dec = nn.Identity()
        
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
        edge_attr = self.enc_edge(edge_attr)


        for i in range(self.nbLayers):
            y = y+self.layerNormList[i](self.GNNLayers[i](y, edge_index, edge_attr))
        
        y = self.dec(y)

        return y
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(d = None):
    if d is None:
        d = GNN_CFG
    
    print(">>>>>>>>>>> Loading Compex model")
    net = deepGNN(d)

    return net