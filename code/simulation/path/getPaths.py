import os
import json
import random


ListDatasets = ['/master/code/simulation/data/mew_0_001_noisy',
                '/master/code/simulation/data/mew_0_001_normal', 
                '/master/code/simulation/data/mew_0_01_noisy', 
                '/master/code/simulation/data/mew_0_01_normal', 
               '/master/code/simulation/data/smallest_0_001_normal']


outputFiles = ['/master/code/simulation/path/mew_0_001_noisy.json',
                '/master/code/simulation/path/mew_0_001_normal.json', 
                '/master/code/simulation/path/mew_0_01_noisy.json', 
                '/master/code/simulation/path/mew_0_01_normal.json', 
               '/master/code/simulation/path/smallest_0_001_normal.json',]

def readJson(filePath:str):
    """
    Function to read json 
    """
    
    with open(filePath, 'r') as f:
        data = json.load(f)
    return data


def writeJson(data, filePath):
    """
    Function to write json 
    """
    with open(filePath, 'w') as f:
        json.dump(data, f, indent=2)

        
        

def getPaths(path, jsonFile):
    
    # append paths
    
    learningPath = os.path.join(path, 'training/torch_file')
    validationPath = os.path.join(path, 'validation/torch_file')
    testPath = os.path.join(path, 'test/torch_file')
    
    # create list of paths
    
    trainingList = []
    validationList = []
    testList = []
    
    # update lists
    
    for root, dirs, files in os.walk(learningPath):
        for file in files:
            trainingList.append(os.path.join(learningPath,file))

    for root, dirs, files in os.walk(validationPath):
        for file in files:
            validationList.append(os.path.join(validationPath,file))

    for root, dirs, files in os.walk(testPath):
        for file in files:
            testList.append(os.path.join(testPath,file))
            
    
    # shuffling data
    
    random.shuffle(trainingList)
    random.shuffle(validationList)
    random.shuffle(testList)
    
    
    # create the dict
    
    d = {}
    d['training'] = trainingList
    d['validation'] = validationList
    d['test'] = testList
    
    
    writeJson(d, jsonFile)
    
    
    # verbose
    
    print(f"Created json ({jsonFile}) !!!")
    print(len(trainingList))
    print(len(validationList))
    print(len(testList))
    
    
    return d




def main():
    
    for i in range(len(ListDatasets)):
        getPaths(ListDatasets[i],outputFiles[i] )
        
        
if __name__ == '__main__':
    main()