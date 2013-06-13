#import matplotlib
#matplotlib.use('Qt4Agg')
#import matplotlib.pylab as pylab
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
#from scipy import diag, arange, meshgrid, where
#from numpy.random import multivariate_normal

#from cv2 import *
#from CameraCapture import loadImg,captureImg
"""
for i in range(2):
    captureImg([128,128],"testdata",".bmp",5,100 )
"""
from pybrain.tools.shortcuts        import *
from pybrain.structure              import SigmoidLayer
from pybrain.datasets               import ClassificationDataSet
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.datasets               import ClassificationDataSet
from pybrain.utilities              import percentError
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.structure.modules      import SoftmaxLayer
from CameraCapture                  import loadImg

class NeuralNetwork:
    def __init__(self,size,hidden,iterations,proportion):
        self.width = size[0]
        self.height = size[1]
        self.loadClasses("classes.txt")
        self.alldata = ClassificationDataSet(self.width*self.height,1,nb_classes=self.number_classes)
        
        klass = 0
        for k in range(40):
            if k < 4:
                continue
            if k < 40:
                if k >= (klass+1)*8:
                    klass += 1
                    #print k,klass
            elif k == 40:
                klass = 0
            else:
                if (k-40) >= (klass+1)*8:
                    klass += 1
                    #print k,klass 
            img = loadImg("testdata"+str(k)+".bmp",scale = size)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    data.append(img[i,j])
            self.alldata.addSample(data,[klass])
        if proportion != 0:
            tstdata, trndata = self.alldata.splitWithProportion( 0.01*proportion )
        else:
            trndata = self.alldata
        trndata._convertToOneOfMany( )
        if proportion != 0:
            tstdata._convertToOneOfMany( )
        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim
        self.fnn = buildNetwork( trndata.indim, hidden, trndata.outdim, hiddenclass=SigmoidLayer,outclass=SoftmaxLayer )
        self.trainer = BackpropTrainer( self.fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
        for i in range(iterations):
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
                
        dataTest = ClassificationDataSet(size[0]*size[1],1,nb_classes=self.number_classes)

        for k in range(5):
            img = loadImg("testdata"+str(100+k)+".bmp",scale=size)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    data.append(img[i,j])
            dataTest.addSample(data,[k])
            out = self.fnn.activateOnDataset(dataTest)
            print k,out
            print out.argmax(axis=1)    
        out = percentError(self.trainer.testOnClassData(dataset=dataTest),dataTest['class'])
        print "specialTests: " + str(out)
        
    def classify(self,img):
        dataTest = ClassificationDataSet(self.width*self.height,1,self.number_classes)
        dataTest.addSample(img,[0])
        return self.fnn.activateOnDataset(dataTest)
    def loadClasses(self,filename):
        file = open(filename,'r')
        self.classes = []
        for line in file:
            self.classes.append(line)
        self.number_classes = len(self.classes)