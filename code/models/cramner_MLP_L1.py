import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree as deg

from modules import MLP, MLP2


LATENT_SHAPE = 128

# GNN related parameters
EDGES_SHAPE = 3
MESSAGE_SHAPE = 64
HIDDEN_NN_SHAPE = 64


# output
OUTPUT_SHAPE= 2



class GN3(MessagePassing):
    def __init__(self, inputShape:int, messageShape:int, outputShape:int, shapeEdges:int = 7, hiddenShape:int=64, aggr:str='add'):
        super(GN3, self).__init__(aggr=aggr)

        self.inputShape = inputShape
        self.messageShape = messageShape
        self.outputShape = outputShape
        self.hiddenShape = hiddenShape
        
        self.messageMLP = MLP2(shapeEdges, hiddenShape, messageShape)
        self.norm = torch.nn.LayerNorm(messageShape)
        
        self.updateMLP = MLP(messageShape + inputShape, outputShape)

        self.messageMLP_sup = MLP(messageShape, messageShape)
    
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):
        # Identify isolated nodes
        degree = deg(edge_index[0], num_nodes=x.size(0))
        isolated_nodes_mask = degree == 0
        
        # Propagate messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, x=x)
        
        # Set aggregated output for isolated nodes to null vector
        if isolated_nodes_mask.any():
            null_vector = torch.zeros_like(out[isolated_nodes_mask])
            out[isolated_nodes_mask] = null_vector
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """
        
        y = self.norm(self.messageMLP(edge_attr))
        y = self.messageMLP_sup(y)
        return y
    
    def update(self, aggr_out:torch.tensor, x:torch.tensor):
        """ 
        Function to update all the nodes after the aggregation
        
        Args:
        -----
        - `aggr_out`: result after the aggregation [# Nodes, messageShape]
        - `x`: current node [1, inputShape]
        """

        xVal = torch.cat([x, aggr_out], dim=-1)
        return self.updateMLP(xVal) 
    
    

       
 

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
        self.enc_sup = MLP(latentShape, latentShape)
        
        # GNN
        self.GNN = GN3(latentShape, messageShape, outShape, edge_shape, hiddenGN)
        
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
        y = self.applyEnc(x)
        
        # gnn part
        y = self.GNN(y, edge_index, edge_attr)                                     # [#Nodes, outSHpae]

        return y


    def applyEnc(self, x):
        y = self.enc(x)                                                    
        y = self.norm1(y)
        y = self.enc_sup(y)

        return y
    

    def L1Reg(self, graph):
        atr = graph.edge_attr

        messages = self.GNN.message(None, None, atr)

        loss = 0.01 * torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        return loss


def loadNetwork(inputShape, edge_shape = EDGES_SHAPE):
    net = GN_NN(inputShape, LATENT_SHAPE, OUTPUT_SHAPE, MESSAGE_SHAPE,edge_shape, HIDDEN_NN_SHAPE)

    return net