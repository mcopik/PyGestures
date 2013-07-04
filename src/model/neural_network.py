'''
Created on Jun 29, 2013

@author: mcopik
'''
import string
from os import listdir
from os.path import isfile, join

import cv2.cv   as cv
from pybrain.datasets               import ClassificationDataSet

class NeuralNetwork():
    '''
    '''
    DEFAULT_WIDTH = 40
    DEFAULT_HEIGHT = 40
    DEFAULT_HIDDEN_LAYER_SIZE = 900
    DEFAULT_ITERATIONS_NUMBER = 20
    classes = []
    all_data = None
    TRAIN_DATA_PATH = "train_data"
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
            self.classes.append(line)
        f.close()
    def loadTestData(self):
        self.alldata = ClassificationDataSet(self.width*self.height,1,nb_classes=len(self.classes))
        onlyfiles = [ f for f in listdir(self.TRAIN_DATA_PATH) if isfile(join(self.TRAIN_DATA_PATH,f)) ]
        for file in onlyfiles:
            #check if it is our data
            strings = string.split(file,"testdata_")
            if len(strings) != 1 or strings[0] != file:
                strings = string.split(strings[1],"_")
                class_number = int(strings[0])
                img = cv.LoadImageM(file)
                thumbnail = cv.CreateMat(self.DEFAULT_HEIGHT,self.DEFAULT_WIDTH,cv.CV_8UC1)
                cv.Resize(img,thumbnail)
                img = thumbnail
                data = []
                for i in range(img.width):
                    for j in range(img.height):
                        data.append(img[i,j])
                self.alldata.addSample(data,[class_number])

        


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