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
    



