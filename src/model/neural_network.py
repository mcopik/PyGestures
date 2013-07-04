'''
Created on Jun 29, 2013

@author: mcopik
'''
import string
from os import listdir
from os.path import isfile, join

import cv2.cv   as cv
from pybrain.datasets               import ClassificationDataSet
from pybrain.utilities              import percentError
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.structure.modules      import SoftmaxLayer
from pybrain.structure              import SigmoidLayer

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
            
            strings = string.split(file,"train_data_")
            if len(strings) != 1 or strings[0] != file:
                strings = string.split(strings[1],"_")
                class_number = int(strings[0])
                print file,class_number
                img = cv.LoadImageM(join(self.TRAIN_DATA_PATH,file))
                thumbnail = cv.CreateMat(self.DEFAULT_HEIGHT,self.DEFAULT_WIDTH,cv.CV_8UC3)
                cv.Resize(img,thumbnail)
                img = thumbnail
                gray_image = cv.CreateImage(cv.GetSize(img),8,1)
                cv.CvtColor(img,gray_image,cv.CV_RGB2GRAY)
                data = []
                for i in range(img.width):
                    for j in range(img.height):
                        data.append(gray_image[i,j])
                self.alldata.addSample(data,[class_number])
    def trainNetwork(self,proportion = 0):        
        if proportion != 0:
            tstdata, trndata = self.alldata.splitWithProportion( 0.01*proportion )
        else:
            trndata = self.alldata
        trndata._convertToOneOfMany( )
        if proportion != 0:
            tstdata._convertToOneOfMany( )
        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim
        self.fnn = buildNetwork( trndata.indim, self.hidden_layer_size, trndata.outdim, 
                                 hiddenclass=SigmoidLayer,outclass=SoftmaxLayer )
        self.trainer = BackpropTrainer( self.fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
        for i in range(self.iterations_number):
            self.trainer.trainEpochs( 1 )
            trnresult = percentError( self.trainer.testOnClassData(),
                                      trndata['class'] )
            if proportion != 0:
                tstresult = percentError( self.trainer.testOnClassData(
                   dataset=tstdata ), tstdata['class'] )
        
            if proportion != 0:
                print "epoch: %4d" % self.trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult
            else:
                print "epoch: %4d" % self.trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult


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