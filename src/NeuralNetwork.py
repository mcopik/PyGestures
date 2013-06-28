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
import cv2.cv                       as cv
import os
class NeuralNetwork:
    def __init__(self,size,hidden,iterations,proportion):
        self.width = size[0]
        self.height = size[1]
        self.loadClasses("classes.txt")
        self.number_classes = 3
        self.alldata = ClassificationDataSet(self.width*self.height,1,nb_classes=self.number_classes)
        
        cv.NamedWindow("Captured image")
        klass = 0
        """for k in range(40):
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
            #img = loadImg("testdata"+str(k)+".bmp",scale = size)
            img = cv.LoadImage("testdata"+str(k)+".bmp")
            thumbnail = cv.CreateMat(size[0],size[1],cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            img = thumbnail
            gray_image = cv.CreateImage(cv.GetSize(img), 8, 1)
            cv.CvtColor(img, gray_image, cv.CV_RGB2GRAY)
            df_dx = cv.CreateImage(cv.GetSize(img),8,1)
            #cv.Sobel(gray_image, df_dx, 1, 0)
            
            cv.Canny(gray_image,df_dx,50,150,3)
            
            cv.ShowImage("Captured image", df_dx)
            cv.WaitKey(1000)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    #data.append(img[i,j])
                    data.append(df_dx[i,j])
            self.alldata.addSample(data,[klass])"""
        for i in range(18):
            
            str1 = "frame-00"
            if(i < 10):
                str1 += "0"+str(i)
            else:
                str1 += str(i)
            str1 += ".jpg"
            str1 = os.path.join("0000\\",str1)
            print str1
            img = cv.LoadImage(str1)
            
            thumbnail = cv.CreateMat(size[0],size[1],cv.CV_8UC3)
            cv.Resize(img,thumbnail)
            img = thumbnail
            gray_image = cv.CreateImage(cv.GetSize(img),8,1)
            cv.CvtColor(img,gray_image,cv.CV_RGB2GRAY)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    data.append(gray_image[i,j])
            self.alldata.addSample(data,[0])
        for i in range(18):
        
            str1 = "0001/frame-00"
            if(i < 10):
                str1 += "0"+str(i)
            else:
                str1 += str(i)
            str1 += ".jpg"
                
            img = cv.LoadImage(str1)
            thumbnail = cv.CreateMat(size[0],size[1],cv.CV_8UC3)
            cv.Resize(img,thumbnail)
            img = thumbnail
            gray_image = cv.CreateImage(cv.GetSize(img),8,1)
            cv.CvtColor(img,gray_image,cv.CV_RGB2GRAY)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    data.append(gray_image[i,j])
            self.alldata.addSample(data,[1])
        for i in range(18):
        
            str1 = "0002/frame-00"
            if(i < 10):
                str1 += "0"+str(i)
            else:
                str1 += str(i)
            str1 += ".jpg"
                
            img = cv.LoadImage(str1)
            thumbnail = cv.CreateMat(size[0],size[1],cv.CV_8UC3)
            cv.Resize(img,thumbnail)
            img = thumbnail
            gray_image = cv.CreateImage(cv.GetSize(img),8,1)
            cv.CvtColor(img,gray_image,cv.CV_RGB2GRAY)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    data.append(gray_image[i,j])
            self.alldata.addSample(data,[2])    
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
            #img = loadImg("testdata"+str(100+k)+".bmp",scale=size)
            img = cv.LoadImageM("testdata"+str(100+k)+".bmp")
            thumbnail = cv.CreateMat(size[0],size[1],cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            img = thumbnail
            gray_image = cv.CreateImage(cv.GetSize(img), 8, 1)
            cv.CvtColor(img, gray_image, cv.CV_RGB2GRAY)
            df_dx = cv.CreateImage(cv.GetSize(img),8,1)
            cv.Smooth(gray_image,df_dx,cv.CV_GAUSSIAN,5,5)
            cv.Canny(df_dx,gray_image,50,150,3)
            data = []
            for i in range(img.width):
                for j in range(img.height):
                    #data.append(img[i,j])
                    data.append(gray_image[i,j])
            dataTest.addSample(data,[k])
            out = self.fnn.activateOnDataset(dataTest)
            print k,out
            print out.argmax(axis=1)    
        out = percentError(self.trainer.testOnClassData(dataset=dataTest),dataTest['class'])
        #print "specialTests: " + str(out)cv.CreateMat(size[0],size[1],cv.CV_8UC3)
        
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