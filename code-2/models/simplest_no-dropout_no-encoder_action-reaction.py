import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 5
MESSAGE_SHAPE = 128
HIDDEN_NN_SHAPE = 128


# output
OUTPUT_SHAPE= 2


INTERACTION_CFG = {
    "input_shape": 8,
    "edges_shape": 5,
    "output_shape": 2,
    "MLP_message": {
        "hidden_shape": 128,
        "message_shape": 128,
        "dropout": "no"
    },
    "MLP_update": {
        "hidden_shape": 128,
        "dropout": "no"
    },
    "regularization": {
        "name": "l1",
        "scaler": 0.01
    }
}




def findOppositeIndices(inds):
    
    device = inds.device
    
    inds = inds.cpu()

    inverted_edges = inds[[1, 0], :]

    a = inds.unsqueeze(2) == inverted_edges.unsqueeze(1)

    b = a.all(dim = 0)

    matching_indices = torch.nonzero(b, as_tuple=False)
    
    res = matching_indices[:, 1].to(device)

    return res



class MLP(nn.Module):
    """ 
    linearly growing size
    """
    def __init__(self, inputShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP, self).__init__()

        self.inputShape = inputShape
        self.outputShape = outputShape

        #assert dropout == 'no'

        self.delta = (inputShape - outputShape) // 3
        dim1 = inputShape - self.delta
        dim2 = dim1 - self.delta

        self.mlp = nn.Sequential(
            nn.Linear(inputShape, dim1),
            nn.LeakyReLU(),
            nn.Linear(dim1, dim2),
            #nn.ELU(),
            nn.LeakyReLU(),
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
                layer.bias.data.fill_(0.01)
                
 

class MLP2(nn.Module):
    """
    constant size
    """
    def __init__(self, inputShape:int, latentShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP2, self).__init__()

        self.inputShape = inputShape
        self.latentShape = latentShape
        self.outputShape = outputShape

        #assert dropout == 'no'

        self.mlp = nn.Sequential(
            nn.Linear(inputShape, latentShape),
            nn.LeakyReLU(),
            nn.Linear(latentShape, latentShape),
            nn.LeakyReLU(),
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
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, index_all = edge_index, x=x)
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor, index_all: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """

        mask = index_all[0, :] < index_all[1, :]

        inds_i_j = torch.nonzero(mask, as_tuple=False).squeeze()

        inds_j_i = findOppositeIndices(index_all)


        inds_j_i = inds_j_i[inds_i_j]
        
        y = self.messageMLP(edge_attr)

        y[inds_j_i, :] = - y[inds_i_j, :]
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
    

    def L1Reg(self, graph):
        atr = graph.edge_attr

        messages = self.GNN.message(None, None, atr)

        loss = self.scaler_regu * torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        return loss


def loadNetwork(d = None):

    if d is None:
        d = INTERACTION_CFG

    print('>>>> loading simplest')
    print('INFO >>> with NO encoder')
    print('INFO >>> with NO dropout')
    print('INFO >>> updated BIASES')
    print('INFO>>> adeed action-reaction')
    net = Simplest(d)

    return net