import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import itertools
from torch_geometric.utils import sort_edge_index


GAM_CFG = {
    "input_shape": 8,
    "edges_shape": 5,
    "output_shape": 2,
    "MLP_message": {
        "hidden_shape": 128,
        "dropout": "no"
    },
    "MLP_update": {
        "hidden_shape": 128,
        "dropout": "no"
    },
    "Basis": {
        "basis": "poly",
        "degree": 2,
        "nDim": 128,
    },
    "regularization": {
        "name": "l1",
        "scaler": 0.01# changed
    }
}


def unique_ids(indices):
    return indices[0] * indices.shape[1] + indices[1]


def compute_num_products(num_variables, degree):
    """
    Compute the number of polynomial products for a given number of variables and degree.
    
    Args:
        num_variables (int): The number of variables.
        degree (int): The maximum degree of the polynomial terms.

    Returns:
        int: The total number of polynomial products.
    """
    num_products = 0
    for d in range(1, degree + 1):
        num_products += sum(1 for _ in itertools.combinations_with_replacement(range(num_variables), d))
    return num_products


def compute_products_batch(tensor, degree):
    """
    Compute all polynomial products up to a given degree for a batch of input tensors using PyTorch operations.
    
    Args:
        tensor (torch.Tensor): The input tensor with shape [batch_size, num_variables].
        degree (int): The maximum degree of the polynomial terms.

    Returns:
        torch.Tensor: A tensor containing all polynomial products for the batch, shape [batch_size, num_products].
    """
    batch_size, num_variables = tensor.shape
    num_products = compute_num_products(num_variables, degree)
    products = torch.empty(batch_size, num_products, device=tensor.device)

    idx = 0
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(num_variables), d):
            product = torch.prod(tensor[:, list(combo)], dim=1)
            products[:, idx] = product
            idx += 1

    return products


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
                layer.bias.data.fill_(0.01)
                              
                
class GN_edge_GAM(MessagePassing):
    """ 
    Message passing neural network in which the message passing
    only considers the edges features
    """
    def __init__(self, d, aggr:str='add'):
        
        super(GN_edge_GAM, self).__init__(aggr=aggr)

        self.inputShape = d['input_shape']
        self.edgeShape = d['edges_shape']
        self.outputShape = d['output_shape']
        
        self.degreePoly = d['Basis']['degree']
        self.basis = d['Basis']['basis']
        self.nbDim = d['Basis']['nDim']
        self.messageShape = compute_num_products(d['edges_shape'], d['Basis']['degree']) * d['Basis']['nDim']
        
        self.hiddenShape = d['MLP_message']['hidden_shape']
        
        self.messageMLP = MLP2(inputShape = self.edgeShape, 
                               latentShape = self.hiddenShape, 
                               outputShape = self.messageShape, 
                               dropout = d['MLP_message']['dropout'])
        
        self.norm = torch.nn.LayerNorm(self.nbDim + self.inputShape)
        
        self.updateMLP = MLP(inputShape = self.nbDim + self.inputShape, 
                             outputShape = self.outputShape,
                             dropout = d['MLP_update']['dropout'])

    
    
    def forward(self, x:torch.tensor, edge_index:torch.tensor, edge_attr: torch.tensor):

        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), edge_attr=edge_attr, index_full = edge_index, x=x)
        
        return out
      
    def message(self, x_i:torch.tensor, x_j:torch.tensor, edge_attr: torch.tensor, index_full:torch.tensor):
        """ 
        Perfomrs the message passing in the graph neural network
        
        Args:
        -----
        - `x_i`: tensor associated to node i
        - `x_j`: tensor associated to node j
        """

        device = index_full.device

        ij_inds = unique_ids(index_full)
        ji_edge_index = torch.flip(index_full, [0])
        ji_inds = unique_ids(ji_edge_index)

        ji_vect_temp = torch.searchsorted(ij_inds, ji_inds).to(device)

        mask = index_full[0, :] < index_full[1, :]

        inds_i = torch.arange(index_full.shape[1], device = device)[mask]
        inds_j = ji_vect_temp[inds_i]

        # compute the products
        prods = compute_products_batch(edge_attr, self.degreePoly)
        shape = prods.shape  #[N, nb_prods]
        prods = prods.repeat(1, self.nbDim).reshape(self.nbDim * shape[0], shape[1])        # [N, nb_prods * nb_dim] -> [N*nb_dim, nb_prod]

        # obtain the weights
        weights = self.messageMLP(edge_attr).reshape(prods.shape)

        res = torch.sum(weights * prods, axis = -1)

        res = res.reshape(int(res.shape[0]/self.nbDim), self.nbDim)

        res[inds_j, :] = -res[inds_i, :]

        return res
    
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

 
class GAM_GNN(nn.Module):
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
        self.GNN = GN_edge_GAM(d)
        
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
        inds = graph.edge_index

        prods = compute_products_batch(atr, self.GNN.degreePoly)
        shape = prods.shape
        prods = prods.repeat(1, self.GNN.nbDim).reshape(self.GNN.nbDim * shape[0], shape[1])

        # obtain the weights
        weights = self.GNN.messageMLP(atr).reshape(prods.shape)

        loss0 = self.scaler_regu * torch.sum(torch.abs(weights)) / graph.edge_index[0, :].shape[0]

        messages = self.GNN.message(None, None, atr, inds)

        l = torch.sum(torch.abs(messages)) / graph.edge_index[0, :].shape[0]

        loss1 = self.scaler_regu * l

        loss = loss0 + loss1

        return loss 
    
    """
    def get_weights(self, graph):
        atr = graph.edge_attr

        prods = compute_products_batch(atr, self.GNN.degreePoly)
        shape = prods.shape
        prods = prods.repeat(1, self.GNN.nbDim).reshape(self.GNN.nbDim * shape[0], shape[1])

        # obtain the weights
        weights = self.GNN.messageMLP(atr)

        return weights, prods.shape
    """

    def get_weights(self, graph):

        
        index_full = graph.edge_index
        edge_attr = graph.edge_attr
        device = index_full.device

        ij_inds = unique_ids(index_full)
        ji_edge_index = torch.flip(index_full, [0])
        ji_inds = unique_ids(ji_edge_index)

        ji_vect_temp = torch.searchsorted(ij_inds, ji_inds).to(device)

        mask = index_full[0, :] < index_full[1, :]

        inds_i = torch.arange(index_full.shape[1], device = device)[mask]
        inds_j = ji_vect_temp[inds_i]

        # compute the products
        prods = compute_products_batch(edge_attr, self.GNN.degreePoly)
        shape = prods.shape  #[N, nb_prods]
        prods = prods.repeat(1, self.GNN.nbDim).reshape(self.GNN.nbDim * shape[0], shape[1])        # [N, nb_prods * nb_dim] -> [N*nb_dim, nb_prod]

        # obtain the weights
        weights = self.GNN.messageMLP(edge_attr).reshape(prods.shape)

        res = weights * prods

        res = res.reshape(-1, self.GNN.messageShape)

        #res[inds_j, :] = -res[inds_i, :]

        return res,prods.shape


    def message(self, graph):
        atr = graph.edge_attr
        inds = graph.edge_index

        inds, atr = sort_edge_index(inds, atr)

        messages = self.GNN.message(None, None, atr, inds)

        return messages

def loadNetwork(d = None):
    if d is None:
        d = GAM_CFG
    print('>>>> loading simplest')
    print('INFO >>> with NO encoder')
    print('INFO >>> with NO dropout')
    net = GAM_GNN(d)

    return net
