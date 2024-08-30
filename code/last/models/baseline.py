import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 5
MESSAGE_SHAPE = 2
HIDDEN_NN_SHAPE = 128


# output
OUTPUT_SHAPE= 2


BASELINE_CFG = {
    "input_shape": 8,
    "edges_shape": 5,
    "output_shape": 2,
    "MLP_message": {
        "hidden_shape": 128,
        "message_shape": 2,
        "dropout": "no"
    },
    "MLP_update": {
        "hidden_shape": 128,
        "dropout": "no"
    },
    "regularization": {
        "name": "NO",
        "scaler": 0.0
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

    
 

class Simplest(nn.Module):
    def __init__(self,d):
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

        self.regu = d['regularization']['name']
        self.scaler_regu = d['regularization']['scaler']
        
        
        # GNN
        self.GNN = GN_edge(d)
        
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
        #y = self.applyEnc(x)
        
        # gnn part
        y = self.GNN(x, edge_index, edge_attr)                                     # [#Nodes, outSHpae]

        return y


    def applyEnc(self, x):                                                    

        return x
    

    def message(self, graph):
        atr = graph.edge_attr

        return self.GNN.message(None, None, atr)
    

    def L1Reg(self, graph):

        return 0


def loadNetwork(d = None):

    if d is None:
        d = BASELINE_CFG
    print('>>>> loading baseline')
    print('INFO >>> Forcing D = 2')
    print('INFO >>> with NO encoder')
    print('INFO >>> with NO dropout')
    net = Simplest(d)

    return net