'''
Created on Jun 28, 2013

@author: mcopik
'''
import time

import numpy as np
import cv2.cv as cv
from PyQt4.QtGui import QMainWindow,QGridLayout,QWidget,QPushButton,QSlider
from PyQt4.QtCore import SIGNAL,Qt


from video_widget import VideoWidget
from model.image_processing import ImageProcessing
from model.neural_network import NeuralNetwork

class MainWindow(object):
    '''
    Main window in View.
    '''
    MAIN_WINDOW_TITLE = "PyGestures"
    def __init__(self):
        '''
        Constructor
        '''
        self.mainWindow = QMainWindow();
        self.mainWindow.setWindowTitle(self.MAIN_WINDOW_TITLE)
        self.mainWindowSignals = MainWindowSignals(self);
        self._createWidgets();
        self._connectSignals();
        self.mainWindow.setCentralWidget(self.mainWidget);
        self.mainWindow.show();
        
    def _createWidgets(self):
        '''
        '''
        self.mainWidget = QWidget(self.mainWindow)
        self.videoWidget = VideoWidget()
        self.scaleButton = QPushButton("Scale capture",self.mainWindow)
        self.trainNetworkButton = QPushButton("Train network",self.mainWindow)
        self.captureButton = QPushButton("Capture",self.mainWindow)
        self.classifyButton = QPushButton("Classify",self.mainWindow)
        self.layout = QGridLayout();
        self.layout.addWidget(self.videoWidget,0,0)
        self.layout.addWidget(self.scaleButton,1,0)
        self.mainWidget.setLayout(self.layout)
        self.cbMinSlider = QSlider(Qt.Horizontal, self.mainWindow)
        self.layout.addWidget(self.cbMinSlider,2,0)
        self.cbMaxSlider = QSlider(Qt.Horizontal, self.mainWindow)
        self.layout.addWidget(self.cbMaxSlider,3,0)
        self.crMinSlider = QSlider(Qt.Horizontal, self.mainWindow)
        self.layout.addWidget(self.crMinSlider,4,0)
        self.crMaxSlider = QSlider(Qt.Horizontal, self.mainWindow)
        self.layout.addWidget(self.crMaxSlider,5,0)
        self.layout.addWidget(self.captureButton,6,0)
        self.layout.addWidget(self.trainNetworkButton,7,0)
        self.layout.addWidget(self.classifyButton,8,0)
        #self.cbMinSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        #self.cbMinSlider.setGeometry(30, 40, 100, 30)
        self.cbMinSlider.setValue(80)
        self.cbMinSlider.setRange(0,255)
        self.cbMinSlider.valueChanged[int].connect(self.mainWindowSignals.changedValueCbMin)
        self.cbMaxSlider.setRange(0,255)
        self.cbMaxSlider.setValue(135)
        self.cbMaxSlider.valueChanged[int].connect(self.mainWindowSignals.changedValueCbMax)
        self.crMinSlider.setRange(0,255)
        self.crMinSlider.setValue(130)
        self.crMinSlider.valueChanged[int].connect(self.mainWindowSignals.changedValueCrMin)
        self.crMaxSlider.setRange(0,255)
        self.crMaxSlider.setValue(180)
        self.crMaxSlider.valueChanged[int].connect(self.mainWindowSignals.changedValueCrMax)
    def _connectSignals(self):
        '''
        '''
        self.mainWindowSignals.camera = self.videoWidget._capture
        self.mainWindow.connect(self.scaleButton,SIGNAL("clicked()"),self.mainWindowSignals.prepareCapture)
        self.mainWindow.connect(self.captureButton,SIGNAL("clicked()"),self.mainWindowSignals.captureImage)
        self.mainWindow.connect(self.trainNetworkButton,SIGNAL("clicked()"),self.mainWindowSignals.trainNetwork)
        self.mainWindow.connect(self.classifyButton,SIGNAL("clicked()"),self.mainWindowSignals.captureImage)

class MainWindowSignals():
    '''
    '''
    cb_min = 80
    cb_max = 135
    cr_min = 130
    cr_max = 180
    camera = None
    def __init__(self,main_window):
        self.main_window = main_window
    def trainNetworkx(self):
        process = ImageProcessing()
        #process.scale_width = 40
        #process.scale_height = 40
        #func = process.medianFilter(process.combinedSkinDetection())
        func2 = process.medianFilter(process.yCbCrSkinDetection())
        #func3 = process.medianFilter2(process.processContour(process.medianFilter(process.yCbCrSkinDetection())))
        #func4 = process.combinedSkinDetection(process.medianFilter())
        #for i in range(79):
            
        #    img = cv.LoadImageM("testdata%d.bmp"%i)
        #    out = func(img)
       #     cv.SaveImage("testdata%dcontour.png"%i,out)
        #    cv.NamedWindow("Captured image")
        #    cv.ShowImage("Captured image", out)#cv.fromarray(out))
        
        start_time = time.clock()
        cur_time = 0
        cv.NamedWindow("Captured image", 1)
        while True:
            start_time = time.clock()
            img = cv.QueryFrame(self.main_window.videoWidget._capture)
            #img = cv.LoadImage("frame-0000.jpg")
            thumbnail = cv.CreateMat(240, 320, cv.CV_8UC3)
            ycr = cv.CreateMat(240, 320, cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            #cv.SaveImage("img.png", thumbnail)
            cv.CvtColor(thumbnail, ycr, cv.CV_RGB2YCrCb)
            img = thumbnail
            #image = cv.CreateImage(cv.GetSize(img),8,len(img[0,0]))
            #cv.Copy(img, image)
            for i in range(img.width):
                for j in range(img.height):
                    val = ycr[j,i]
                    if not (val[1] > 80 and val[1] < 135 and
                            val[2] > 130 and val[2] < 180):
                        img[j,i] = [0,0,0]
            #cv.SaveImage("img_1.png", img)
            '''
            img = image
            vec = np.array([16,128,128])
            mat = np.array([[65.481,128.553,24.966],
                            [-37.797,-74.203,112],
                            [112,-93.786,-18.214]])

            for i in range(img.width):
                for j in range(img.height):
                    rgb = np.array([img[j,i][2],img[j,i][1],img[j,i][0]])
                    val = mat.dot(rgb)
                    #val = 1/256(mat*rgb + vec)
                    val = val/256 + vec
                    if not (val[1] > 80 and val[1] < 135 and
                            val[2] > 130 and val[2] < 180):
                        img[j,i] = [0,0,0]
            '''
            cv.SaveImage("img_2.png", img)
            cv.ShowImage("Captured image", img)
            key = cv.WaitKey(10)
            cur_time = cur_time + 1;
            print time.clock() - start_time
            #break
        
        cur_time = time.clock() - start_time
        print cur_time
    def trainHistogramNetwork(self):
        pass
    def prepareCapture(self,size=[320,240]):
        start_time = time.clock()
        cv.NamedWindow("Captured image", 1)
        while True:
            start_time = time.clock()
            img = cv.QueryFrame(self.camera)
            thumbnail = cv.CreateMat(size[1], size[0], cv.CV_8UC3)
            ycr = cv.CreateMat(size[1],size[0], cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            cv.CvtColor(thumbnail, ycr, cv.CV_RGB2YCrCb)
            img = thumbnail
            #print self.cb_max,self.cb_min
            for i in range(img.width):
                for j in range(img.height):
                    val = ycr[j,i]
                    if not (val[1] > self.cb_min and val[1] < self.cb_max and
                            val[2] > self.cr_min and val[2] < self.cr_max):
                        img[j,i] = [0,0,0]
            cv.ShowImage("Captured image", img)
            key = cv.WaitKey(10)
            #print "Compute time %f s"%(time.clock() - start_time)
    def captureImage(self,size=[320,240]):
        cv.NamedWindow("Contours", 1)
        process = ImageProcessing()
        process.cb_min = self.cb_min
        process.cb_max = self.cb_max
        process.cr_min = self.cr_min
        process.cr_max = self.cr_max
        func = process.processContour(process.medianFilter(process.yCbCrSkinDetection()))
        img = cv.QueryFrame(self.camera)
        thumbnail = cv.CreateMat(size[1], size[0], cv.CV_8UC3)
        cv.Resize(img, thumbnail)
        img = thumbnail
        img = func(img)
        cv.ShowImage("Contours",img)
        class_number = self.nn.classifyImg(img)
    def changedValueCbMin(self,x):
        self.cb_min = x
    def changedValueCbMax(self,x):
        self.cb_max = x    
    def changedValueCrMin(self,x):
        self.cr_min = x
        print x
    def changedValueCrMax(self,x):
        self.cr_max = x
    def trainNetwork(self):
        self.nn = NeuralNetwork([40,40])
        self.nn.loadClassesFromFile("train_data\\classes.txt")
        self.nn.loadTestData()
        self.nn.trainNetwork(10)