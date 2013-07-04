'''
Created on Jun 28, 2013

@author: mcopik
'''
import cv2.cv as cv
from PyQt4.QtGui import QMainWindow,QGridLayout,QWidget,QPushButton
from PyQt4.QtCore import SIGNAL

from video_widget import VideoWidget
from model.image_processing import ImageProcessing

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
        
    def _connectSignals(self):
        '''
        '''
        self.mainWindow.connect(self.trainNetworkButton,SIGNAL("clicked()"),self.mainWindowSignals.trainNetwork)

class MainWindowSignals():
    '''
    '''
    def __init__(self,main_window):
        self.main_window = main_window
    def trainNetwork(self):
        process = ImageProcessing()
        process.scale_width = 40
        process.scale_height = 40
        func = process.processContour(process.medianFilter(process.yCbCrSkinDetection()))
        func2 = process.processContour(process.medianFilter(process.yCbCrSkinDetection()))
        func3 = process.medianFilter2(process.processContour(process.medianFilter(process.yCbCrSkinDetection())))
        func4 = process.combinedSkinDetection(process.medianFilter())
        for i in range(1):
            
            img = cv.LoadImageM("testdata%d.bmp"%i)
            out = func(img)
            cv.Save("testdata%dcontour.png"%i,out)
            cv.NamedWindow("Captured image")
            cv.ShowImage("Captured image", out)#cv.fromarray(out))
    def trainHistogramNetwork(self):
        