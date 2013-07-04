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
    def __init__(self,size = [DEFAULT_WIDTH,DEFAULT_HEIGHT]):
        self.width = size[0]
        self.height = size[1]
        self.hidden_layer_size = self.DEFAULT_HIDDEN_LAYER_SIZE
        self.iterations_number = self.DEFAULT_ITERATIONS_NUMBER
    def loadClasses(self,classes):
        self.class_number = len(classes)
        self.classes = classes
    def loadClassesFromFile(self,path):
        f = open(path)
        for line in f:
            print line
        f.close()
    def loadTestData(self):
        pass
        


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