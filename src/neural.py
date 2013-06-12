import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pylab as pylab
from cv2 import *
from CameraCapture import loadImg,captureImg
"""
for i in range(2):
    captureImg([128,128],"testdata",".bmp",5,100 )
"""
from pybrain.tools.shortcuts import *
from pybrain.structure import SigmoidLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

alldata = ClassificationDataSet(40*40,1,nb_classes=5)
klass = 0
for k in range(40):
    if k < 40:
        if k >= (klass+1)*8:
            klass += 1
            print k,klass
    elif k == 40:
        klass = 0
    else:
        if (k-40) >= (klass+1)*8:
            klass += 1
            print k,klass 
    img = loadImg("testdata"+str(k)+".bmp",scale = [40,40])
    data = []
    for i in range(img.width):
        for j in range(img.height):
            data.append(img[i,j])
    alldata.addSample(data,[klass])


#means = [(-1,0),(2,4),(3,1)]
#cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
#alldata = ClassificationDataSet(2, 1, nb_classes=3)
#for n in xrange(400):
#    for klass in range(3):
#        input = multivariate_normal(means[klass],cov[klass])
#        alldata.addSample(input, [klass])
tstdata, trndata = alldata.splitWithProportion( 0.25 )
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]
fnn = buildNetwork( trndata.indim, 1000, trndata.outdim, hiddenclass=SigmoidLayer,outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)
#ticks = arange(-3.,6.,0.2)
#X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
#griddata = ClassificationDataSet(128*128,1, nb_classes=3)
#for i in xrange(X.size):
#    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
#griddata._convertToOneOfMany()  # this is still needed to m
for i in range(20):
    trainer.trainEpochs( 1 )
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
dataTest = ClassificationDataSet(40*40,1,nb_classes=2)

for i in range(5):
    img = loadImg("testdata"+str(100+i)+".bmp",scale=[40, 40])
    data = []
    for i in range(img.width):
        for j in range(img.height):
            data.append(img[i,j])
    #if i % 2 == 0:
    dataTest.addSample(data,[i])
    #else:
    #   dataTest.addSample(data,[1])

out = percentError(trainer.testOnClassData(dataset=dataTest),dataTest['class'])
print "error: " + str(out)


 #   out = fnn.activateOnDataset(griddata)
#    out = out.argmax(axis=1)  # the highest output activation gives the class
#    out = out.reshape(X.shape)

#img = captureImg("image_gray.bmp")
#DS = ClassificationDataSet(img.width*img.height,class_labels=["FIRST","SECOND"])
#DS.addSample(img,"FIRST")
#net = buildNetwork(img.width*img.height,100,2,hiddenclass=SigmoidLayer)
#trainer = BackPropTrainer(net,ds)
#trainer.trainUntilConvergence()

#convertImg("image.bmp","image_gray.bmp",ConversionType.AVERAGE)
#captureImg([128,128],"image.bmp")
#captureImg("image.bmp")
#loadImg("image.bmp")
#def save():
#    cv.SaveImage("image.bmp", img)
# initialize the camera
#frame = cv.CaptureFromCAM(0)
#img = cv.QueryFrame(frame)
#cam = VideoCapture(0)   # 0 -> index of camera
#s, img = cam.read()
    # frame captured without any errors
#cv.NamedWindow("cam-test",cv.CV_WINDOW_AUTOSIZE)
#cv.ShowImage("cam-test", img)
#cv.namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
#imshow("cam-test",img)
#cv.WaitKey(0)
#cv.DestroyWindow("cam-test")