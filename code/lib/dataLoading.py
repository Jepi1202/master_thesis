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
import json
import random


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def readJson(filePath):
    
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data


def writeJson(data, filePath):
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=2)


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
    



class DataLoader2(Dataset):
    """ 
    Pytorch geometric dataloader
    """

    def __init__(self, 
                 root: str, 
                 transform:Optional[None]=None, 
                 pre_transform:Optional[None]=None, 
                 pre_filter:Optional[None]=None, 
                 path:str = None, 
                 jsonFile:str = None,
                 mode = 'training'):
        """
        Initialisation of the pytorch geometric dataset

        This dataset contains the one-step transitions used during the training

        Args:
        -----
            - `root` (str): path of the root where data is put when
            - `transform`
            - `pre_transform`
            - `pre_filter`
            - `path`: path to use in order to form the different sets
        """
        super(DataLoader2, self).__init__(root, transform, pre_transform, pre_filter)

        self.mode = mode
        
        if not os.path.exists(jsonFile):
            self.path = path
            
            learningPath = os.path.join(path, 'training/torch_file')
            validationPath = os.path.join(path, 'validation/torch_file')
            testPath = os.path.join(path, 'test/torch_file')
            
            self.trainingList = []
            self.validationList = []
            self.testList = []
            
            for root, dirs, files in os.walk(learningPath):
                for file in files:
                    self.trainingList.append(os.path.join(learningPath,file))
                    
            for root, dirs, files in os.walk(validationPath):
                for file in files:
                    self.validationList.append(os.path.join(validationPath,file))
                    
            for root, dirs, files in os.walk(testPath):
                for file in files:
                    self.testList.append(os.path.join(testPath,file))



            random.shuffle(self.trainingList)
            random.shuffle(self.validationList)
            random.shuffle(self.testList)



            d = {}
            d['training'] = self.trainingList
            d['validation'] = self.validationList
            d['test'] = self.testList

            writeJson(d, jsonFile)
            
            print(f"Created json !!!")
            print(len(self.trainingList))
            print(len(self.validationList))
            print(len(self.testList))


        else:
            jsonFile = readJson(jsonFile)
            self.trainingList = jsonFile['training']
            self.validationList = jsonFile['validation']
            self.testList = jsonFile['test']


        if self.mode == 'training':
            self.pathList =  self.trainingList
        
        if self.mode == 'validation':
            self.pathList = self.validationList
        
        if self.mode == 'test':
            self.pathList = self.testList

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
        if self.mode == 'training':
            return len(self.trainingList)
        
        if self.mode == 'validation':
            return len(self.validationList)
        
        if self.mode == 'test':
            return len(self.testList)

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

                resOutput = torch.from_numpy(np.load(os.path.join(path, file)))

                self.mats.append(resOutput)
             
        self.length = len(self.mats)
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        return self.mats[idx], idx
    



