import yaml


class Normalization:
    def __init__(self):

        self.normalization_type = 'default'

        self.position = {'minPos': -120, 'maxPos': 120}


        self.distance = {'minDistance': 0, 'maxDistance': 6}

        self.speed = {'minSpeed': -15, 'maxSpeed': 15}
        
        self.radius = {'minRad': 0, 'maxRad': 2}


    def to_dict(self):
        return self.__dict__


#########################

class Feature:
    def __init__(self):
        self.distGraph = 6
        self.nbHist = 4
        self.nbRolloutOut = 10
        self.output = 'speed'
        self.inShape = 8
        self.edgeShape = 5

    def to_dict(self):
        return self.__dict__


###########################

class SimulationParameters:
    def __init__(self):
        self.noisy = 0
        self.nMin = 120
        self.nMax = 300
        self.v0 = 60
        self.k = 70
        self.epsilon = 0.5
        self.tau = 3.5
        self.T = 1000
        self.dt = 0.001
        self.threshold = 6
        self.R = 1
        self.boundary = 120


    def to_dict(self):
        return self.__dict__

class Simulation:
    def __init__(self):
        self.nbSimLearning = 1000
        self.nbValidation = 20
        self.nbTest = 10
        self.initialization = 'easy'        # random, easy, circle
        self.initDistance = 7
        self.parameters = SimulationParameters()


    def to_dict(self):
        data = self.__dict__.copy()
        data['parameters'] = self.parameters.to_dict()
        return data


#############################

class DataAugment:
    def __init__(self):
        self.bool = 1
        self.prob = 0.8
        self.stdDeltaPos = 2
        self.stdSpeed = 0.003

    def to_dict(self):
        return self.__dict__

class Loss:
    def __init__(self):
        self.topk = -1
        self.lossScaling = 100
        self.l1Reg = 0.001
        self.lim = 35
        self.lambdaL2Weights = 0.00005


    def to_dict(self):
        return self.__dict__

class SchedulerExp:
    def __init__(self):
        self.gamma = 0.9

    def to_dict(self):
        return self.__dict__

class SchedulerLinear:
    def __init__(self):
        self.size = 1
        self.gamma = 0.5

    def to_dict(self):
        return self.__dict__

class Training:
    def __init__(self):
        self.nbEpoch = 3000
        self.modelName = 'simplest_dropout_no-encoder'
        self.evalModel = 'simplest_drop_no-enc_aug_best.pt'
        self.saveModel = 'simplest_drop_no-enc_aug_latest.pt'
        self.pathData = '/scratch/users/jpierre/mew_0.01_normal_v2'
        self.pathJsonBool = True
        self.wbName = 'Simplest_normal_0.001-lr_0_005_b-32'
        self.rolloutNb = 4
        self.batch = 16
        self.batchSizeVal = 128
        self.lr = 0.005
        self.dataAugment = DataAugment()
        self.loss = Loss()
        self.scheduler = 'exp'
        self.scheduler_exp = SchedulerExp()
        self.scheduler_linear = SchedulerLinear()
        self.frequenceEval = 50
        self.frequenceSave = 5000

    def to_dict(self):
        data = self.__dict__.copy()
        data['dataAugment'] = self.dataAugment.to_dict()
        data['loss'] = self.loss.to_dict()
        data['scheduler_exp'] = self.scheduler_exp.to_dict()
        data['scheduler_linear'] = self.scheduler_linear.to_dict()
        return data

class Config:
    def __init__(self):
        self.normalization = Normalization()
        self.feature = Feature()
        self.simulation = Simulation()
        self.training = Training()

    def to_dict(self):
        return {
            'normalization': self.normalization.to_dict(),
            'feature': self.feature.to_dict(),
            'simulation': self.simulation.to_dict(),
            'training': self.training.to_dict()
        }



def create_yml_2(cfg):

    # updater mais en gros ok
    YAML_RUN = """
    normalization:
    position:
        minPos: -120
        maxPos: 120

    distance:
        minDistance: 0
        maxDistance: 6

    speed:
        minSpeed: -15
        maxSpeed: 15

    radius:
        minRad: 0
        maxRad: 2

        
    feature:
    distGraph: 6          # distance to create the graph
    nbHist: 4             # Number of histogram bins
    nbRolloutOut: 10      # Number of rollouts in the gt outputs
    output: speed         # Options: distance, speed, acceleration
    inShape: 8            # dimension of the input features x
    edgeShape: 5          # dimension of the edges


    simulation:
    nbSimLearning: 1000                       # number of simulation in the training set
    nbValidation: 20                          # number of simuation in the validation set
    nbTest: 10                                # number of simulation in the test set
    initialization: 'easy'                    # Options: random, easy, circle
    initDistance: 7                           # distnace option for the grid and simple initialization
    
    parameters:               
        noisy: 0                                # 0: not noisy, 1: noisy      # 
        nMin: 120                               # min number of cells    
        nMax: 300                               # max number of cells
        v0: 60                                  # active force constant
        k: 70                                   # 
        epsilon: 0.5                            #
        tau: 3.5                                #
        T: 1000                                 #
        dt: 0.001                               #
        threshold: 6                            #
        R: 1                                    #
        boundary: 120                           #

        
    training:
    nbEpoch: 3000                                                 # number of maximum epoch
    modelName: 'simplest_dropout_no-encoder'                      # name of the model
    evalModel: simplest_drop_no-enc_aug_best.pt                   # name of the saved (best) evaluation model
    saveModel: simplest_drop_no-enc_aug_latest.pt                 # name of the saved (latest) evaluation model
    
    topk: -1

    pathData: '/scratch/users/jpierre/mew_0.001_normal'           # path of the data in the scratch
    pathJsonBool: True                                            # boolean to detect the use of the dictionary of path

    wbName: 'Simplest_normal_0.001-lr_0_005'                      # wandb name of the run
    rolloutNb: 4                                                  # number of rollouts
    batch: 128                                                    # batch size
    batchSizeVal: 128                                             # batch size of the evaluation
    lr: 0.0005                                                    # learning rate
    
    dataAugment:
        bool: 1
        prob: 0.8
        stdDeltaPos: 2
        stdSpeed: 0.003
    
    loss:
        lossScaling: 100
        l1Reg: 0.001
        lim: 35  # Reference: boundary
        lambdaL2Weights: 0.00005
    
    scheduler: 1
    scheduleParams:
        size: 1
        gamma: 0.5

    frequenceEval: 50
    frequenceSave: 5000
    """



    