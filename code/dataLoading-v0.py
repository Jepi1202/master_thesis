import numpy as np
import torch
#import torch.utils.data as torchData
from torch.utils.data import Dataset as torchDataset
from torch_geometric.data import Dataset, Data
#from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
from typing import Optional
import features as ft      


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getGraph(nodeFeatures, yVect, edgeFeaturesList, edgeIndexVect, device:Optional[str] = DEVICE):
    """ 
    Obtain the pytorch_geometric data from associated features

    Args:
    -----
        - ``
        - ``
        - ``
        - ``
        - ``

    Returns:
    --------
        pytorch_geometric data
    """

    # list of data
    dataList = []
    
    for i in range(len(nodeFeatures)):
        x = torch.squeeze(nodeFeatures[i])
        y = torch.squeeze(yVect[i])

        edgeFeat = torch.squeeze(edgeFeaturesList[i])
        edgeInd = torch.squeeze(edgeIndexVect[i])

        dataC = Data(x = x, edge_index = edgeInd, edge_attr = edgeFeat, y = y).to(device)

        dataList.append(dataC)
        
    return dataList


def getPredictions(dataList, net):
    """ 
    
    """
    deltaPos = []

    for i in tqdm(range(len(dataList))):
        
        deltaPos.append(net(dataList[i].x, dataList[i].edge_index, dataList[i].edge_attr).cpu().detach().numpy())
        
    return deltaPos


# necessary ?
def getFeatures(resOutput, deltaPos):
    """ 
    
    """
    feats = ft.getPosSpeedsFeats(resOutput)
    predFeats = np.zeros_like(feats)

    # getting pos

    for t in range(feats.shape[0]-1):
        for n in range(feats.shape[1]):
            predFeats[t+1, n, 0] = feats[t, n, 0] + deltaPos[t][n, 0]   # pos X
            predFeats[t+1, n, 1] = feats[t, n, 1] + deltaPos[t][n, 1]   # pos Y

            predFeats[t+1, n, 2] = deltaPos[t][n, 0]                    # speed X
            predFeats[t+1, n, 3] = deltaPos[t][n, 1]                    # speed Y



    feats = feats[1:, :, :]
    predFeats = predFeats[1:, :, :]
    
    return feats, predFeats



class DataGraph(Dataset):
    """ 
    Pytorch geometric dataloader
    """

    def __init__(self, root: str, transform:Optional[None]=None, pre_transform:Optional[None]=None, pre_filter:Optional[None]=None, path:str = None):
        """
        Initialisation of the pytorch geometric dataset

        This dataset contains the one-step transitions used during the training

        Args:
        -----
            - `root` (str): path of the root where data is put when
            - `transform`
            - `pre_transform`
            - `pre_filter`
            - `path`
        """
        super(DataGraph, self).__init__(root, transform, pre_transform, pre_filter)
        
        if path is not None:
            self.path = path
            
            self.pathList = []
            
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    self.pathList.append(os.path.join(self.path,file))

    @property
    def raw_file_names(self):
        return None

    @property
    def processed_file_names(self):
        return None

    def _download(self):
        pass
    

    def _process(self, pathlist):
        pass
                    
    def len(self):
        return len(self.pathList)

    def get(self, idx):
        data = torch.load(self.pathList[idx])
        
        return data, idx
    


class simLoader(torchDataset):
    """
    Pytorch dataset to display the tuples for learning
    """

    def __init__(self, path:list[str], lim:Optional[int] = None):
        """ 
        Args:
        -----
            - `path` (str): ...
            - `lim``(Otional, int): ...
        
        """
        
        if lim is not None:
            limitation = lim
        else:
            limitation = float('inf')
        

        self.path = path
        self.limitation = limitation
        

        self.pathLists = []
        self.featuresMat = []
        self.y = []
        self.edge_attr = []
        self.edge_ind = []

        for root, dirs, files in tqdm(os.walk(path)):
            for file in files:
                if len(self.pathLists) > limitation:
                    break
                
                if file.startswith("output"):
                    self.pathLists.append(os.path.join(path, file))
                    
                    #resOutput = np.load(file)
                    a = np.load(os.path.join(path, file), allow_pickle=True)
                    resOutput = a.item()['resOutput']
                    params = a.item()['paramList']
                    
                    mat, y = ft.getFeatures(resOutput, params)
                    edgeIndexVect, edgeFeaturesList = ft.getEdges(resOutput)
                    
                    self.featuresMat.append(mat)
                    self.y.append(y)
                    self.edge_attr.append(edgeFeaturesList)
                    self.edge_ind.append(edgeIndexVect)
                    
                    
                    
                    
        self.length = len(self.pathLists)
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        x = self.featuresMat[idx]
        y = self.y[idx]
        edge_at = self.edge_attr[idx]
        edge_ind = self.edge_ind[idx]
        
        p = self.pathLists[idx]
        a = np.load(os.path.join(p), allow_pickle=True)
        resOutput = a.item()['resOutput']
        

        return x, y, edge_at, edge_ind, resOutput, idx
    
    
    
class simLoader2(torchDataset):
    """
    Pytorch dataset to display the tuples for learning
    """

    def __init__(self, path:list[str]):
        """ 
        Args:
        -----
            - `path` (str): ...
            - `lim``(Otional, int): ...
        
        """
        

        self.path = path
        
        self.mats = []

        for root, dirs, files in tqdm(os.walk(path)):
            for file in files:
                
                if file.startswith("output"):
                    
                    
                    #resOutput = np.load(file)
                    a = np.load(os.path.join(path, file), allow_pickle=True)
                    resOutput = a.item()['resOutput']
                    
                    self.mats.append(resOutput)

                    
                    
                    
                    
        self.length = len(self.mats)
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        return self.mats[idx], idx
    



