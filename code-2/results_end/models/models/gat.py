import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing, GATv2Conv

import os



# load the cfg




LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 5
MESSAGE_SHAPE = 128
HIDDEN_NN_SHAPE = 128



GAT_CFG = {
    "input_shape": 8,
    "edges_shape": 5,
    "output_shape": 2,
    "nb_layers":3,
    "action-reacion": 'no',
    "layer_norm":1,
    "encoder":{
        "presence": "no",
        "out_shape": 128,
        "dropout": "no",
        "latent_edge": 128,
    },
    "layer": {
        "hidden_shape": 128,
        "message_shape": 128,
        "dropout": 0.0,
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
        if dropout != 'no':
            mods.append(nn.Dropout(p=dropout))
        
        mods.append(nn.Linear(dim1, dim2)),
        #nn.ELU(),
        mods.append(nn.LeakyReLU()),
        if dropout != 'no':
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
        if dropout != 'no':
            mods.append(nn.Dropout(p=dropout))

        mods.append(nn.Linear(latentShape, latentShape))
        mods.append(nn.LeakyReLU())
        if dropout != 'no':
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
                
        
 

class GN_NN(nn.Module):
    def __init__(self, d):
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
        self.outShape = d['output_shape']
        self.edge_shape = d['edges_shape']
        
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
        
        # GNN
        self.GNNLayers = torch.nn.ModuleList()
        self.layerNormList = torch.nn.ModuleList()
        self.nb_gnn = d['nb_layers']
        #self.GNN = torch.zeros((10, 128))
        
        for i in range(self.nb_gnn):
            #self.GNNLayers.append(GN(latentShape, messageShape, latentShape, edge_shape, hiddenGN))
            self.GNNLayers.append(GATv2Conv(d['layer']['hidden_shape'], 
                                            d['layer']['hidden_shape'], 
                                            heads=1, 
                                            dropout=d['layer']['dropout'], 
                                            concat=True, 
                                            edge_dim=d['layer']['message_shape'], 
                                            add_self_loops=False, 
                                            fill_value=0.0)
                                )
            
            if d['layer_norm']:
                self.layerNormList.append(torch.nn.LayerNorm(d['layer']['hidden_shape']))
            else:
                self.layerNormList.append(torch.nn.Identity())
    
        if d['encoder']['presence']:
            self.dec =MLP(inputShape = d['layer']['message_shape'], 
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

        # gnn part
        for i in range(self.nb_gnn):
            y = y+self.layerNormList[i](self.GNNLayers[i](y, edge_index, edge_attr))
        
        y = self.dec(y)

        return y
    

    def L1Reg(self, graph):
        

        return 0


def loadNetwork(d = None):
    if d is None:
        d = GAT_CFG
    
    print(">>>>>>>>>>> Loading GaT model")
    net = GN_NN(d)

    return net