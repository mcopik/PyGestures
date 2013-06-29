'''
Created on Jun 29, 2013

@author: mcopik
'''

class NeuralNetwork():
    '''
    '''
    DEFAULT_WIDTH = 40
    DEFAULT_HEIGHT = 40
    DEFAULT_HIDDEN_LAYER_SIZE = 900
    DEFAULT_ITERATIONS_NUMBER = 20
    class_number = 0
    classes = []
    def __init__(self):
        '''
        '''
        self.width = self.DEFAULT_WIDTH
        self.height = self.DEFAULT_HEIGHT
        self.hidden_layer_size = self.DEFAULT_HIDDEN_LAYER_SIZE
        self.iterations_number = self.DEFAULT_ITERATIONS_NUMBER
        
    def loadClasses(self,classes):
        self.class_number = len(classes)
        self.classes = classes
    
        


class ImageNeuralNetwork(NeuralNetwork):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    
class HistogramNeuralNetwork(NeuralNetwork):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        pass