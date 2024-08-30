import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MLP(nn.Module):
    """ 
    linearly growing size
    """
    def __init__(self, inputShape:int, outputShape:int, dropout:float = 0.3):
        super(MLP, self).__init__()

        assert dropout == 'no'

        self.inputShape = inputShape
        self.outputShape = outputShape

        self.delta = (inputShape - outputShape) // 3
        dim1 = inputShape - self.delta
        dim2 = dim1 - self.delta

        self.mlp = nn.Sequential(
            nn.Linear(inputShape, dim1),
            nn.LeakyReLU(),
            nn.Linear(dim1, dim2),
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

        assert dropout == 'no'

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
                layer.bias.data.fill_(0.)
                
                
class GN_edge(MessagePassing):
    """ 
    Message passing neural network in which the message passing
    only considers the edges features
    """
    def __init__(self, d, 
                 inputShape:int, 
                 messageShape:int, 
                 outputShape:int, 
                 shapeEdges:int = 7, 
                 hiddenShape:int=64, 
                 aggr:str='add'):
        super(GN_edge, self).__init__(aggr=aggr)

        self.inputShape = d['input_shape']
        self.messageShape = d['messages_latent']
        self.outputShape = d['output_shape']
        self.edgeShape = d['edges_shape']
        self.MLP_message_shaep = d['MLP_message']
        
        self.messageMLP = MLP2(inputShape = self.edgeShape, 
                               latentShape = d['MLP_message']['hidden_shape'], 
                               outputShape = d['messages_latent'], 
                               dropout = d['MLP_message']['hidden_shape'])
        
        self.norm = torch.nn.LayerNorm(d['messages_latent']+ d['input_shape'])
        
        self.updateMLP = MLP(inputShape = d['messages_latent']+ d['input_shape'], 
                             outputShape = d['output_shape'],
                             dropout = d['MLP_update']['dropout'])
    
    
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
    def __init__(self, d):
        """ 
        Neural network to combine everything
        
        Args:
        -----
        - `d`: dictionnary of shape information
        """
        
        super().__init__()
        
        self.inShape = d['input_shape']
        self.outShape = d['output_shape']
        
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

        loss = 0.01 * torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        return loss
    

def loadNetwork(d):
    print('>>>> loading simplest')
    print('INFO >>> with NO encoder')
    print('INFO >>> with NO dropout')
    net = Simplest(d)

    return net