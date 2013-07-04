'''
Created on Jun 28, 2013

@author: mcopik
'''
import time

import numpy as np
import cv2.cv as cv
from PyQt4.QtGui import QMainWindow,QGridLayout,QWidget,QPushButton,QSlider
from PyQt4.QtCore import SIGNAL
from PyQt4.QtCore import *

from video_widget import VideoWidget
from model.image_processing import ImageProcessing
from Tkconstants import HORIZONTAL

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
        self.trainNetworkButton = QPushButton(self.mainWindow)
        self.layout = QGridLayout();
        self.layout.addWidget(self.videoWidget,0,0)
        self.layout.addWidget(self.trainNetworkButton,1,0)
        self.mainWidget.setLayout(self.layout)
        self.cbMinSlider = QSlider(Qt.Horizontal, self.mainWindow)
        self.layout.addWidget(self.cbMinSlider)
        #self.cbMinSlider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cbMinSlider.setGeometry(30, 40, 100, 30)
        self.cbMinSlider.valueChanged[int].connect(self.mainWindowSignals.changedValue)
    def _connectSignals(self):
        '''
        '''
        self.mainWindowSignals.camera = self.videoWidget._capture
        self.mainWindow.connect(self.trainNetworkButton,SIGNAL("clicked()"),self.mainWindowSignals.prepareCapture)

class MainWindowSignals():
    '''
    '''
    camera = None
    def __init__(self,main_window):
        self.main_window = main_window
    def trainNetwork(self):
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
            for i in range(img.width):
                for j in range(img.height):
                    val = ycr[j,i]
                    if not (val[1] > 80 and val[1] < 135 and
                            val[2] > 130 and val[2] < 180):
                        img[j,i] = [0,0,0]
            cv.ShowImage("Captured image", img)
            key = cv.WaitKey(10)
            print "Compute time %f s"%(time.clock() - start_time)
    def changedValue(self,x):
        print x