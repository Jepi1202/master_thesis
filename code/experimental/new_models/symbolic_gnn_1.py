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


def calculate_interaction(dist, cosine, sine, rad1, rad2, k, epsilon):
    """
    Given the vectors ri and rj, compute the force between them
    """

    dist = dist.unsqueeze(-1)
    cosine = cosine.unsqueeze(-1)
    sine = sine.unsqueeze(-1)
    rad1 = rad1.unsqueeze(-1)
    rad2 = rad2.unsqueeze(-1)

    dist_x = cosine * dist
    dist_y = sine * dist

    rij = torch.hstack((dist_x.reshape(-1, 1), dist_y.reshape(-1, 1)))


    r = dist

    # Combined radius of both particles (assume unit radii for now)
    #bij = 2.0                       # Ri + Rj 
    bij = rad1 + rad2

    force = torch.zeros((dist.shape[0], 2))

    force = torch.where(r < bij*(1 + epsilon), k*(r - bij)*rij/r, -k*(r - bij - 2*epsilon*bij)*rij/r)
    force = torch.where(r >= bij*(1 + 2*epsilon), torch.tensor([0.0, 0.0]), force)


    return force


class MLP(nn.Module):
    """ 
    linearly growing size
    """
    def __init__(self, inputShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP, self).__init__()

        self.inputShape = inputShape
        self.outputShape = outputShape

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
                layer.bias.data.fill_(0.)
                
 

class MLP2(nn.Module):
    """
    constant size
    """
    def __init__(self, inputShape:int, latentShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP2, self).__init__()

        self.inputShape = inputShape
        self.latentShape = latentShape
        self.outputShape = outputShape

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
                layer.bias.data.fill_(0.)
                
                
                
class GN_edge(MessagePassing):
    """ 
    Message passing neural network in which the message passing
    only considers the edges features
    """
    def __init__(self, inputShape:int, message_function, outputShape:int, shapeEdges:int = 7, hiddenShape:int=64, messageShape = 2, aggr:str='add'):
        super(GN_edge, self).__init__(aggr=aggr)

        self.inputShape = inputShape
        self.messageShape = messageShape
        self.outputShape = outputShape
        self.hiddenShape = hiddenShape
        
        self.message_function = message_function
        self.norm = torch.nn.LayerNorm(self.messageShape+ inputShape)
        
        self.updateMLP = MLP(self.messageShape + inputShape, outputShape)
    
    
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
        
        y = self.message_function(edge_attr)
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
        
        
        # GNN
        self.GNN = GN_edge(inShape, messageShape, outShape, edge_shape, hiddenGN)
        
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

        loss = 0.01 * torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        return loss


def loadNetwork(inputShape, edge_shape = EDGES_SHAPE):
    print('>>>> loading simplest')
    print('INFO >>> with NO encoder')
    print('INFO >>> with NO dropout')
    net = Simplest(inputShape, LATENT_SHAPE, OUTPUT_SHAPE, MESSAGE_SHAPE,edge_shape, HIDDEN_NN_SHAPE)

    return net