'''
Created on Jun 28, 2013

@author: mcopik
'''

from PyQt4.QtGui import QMainWindow,QGridLayout,QWidget

from video_widget import VideoWidget

class MainWindow(object):
    '''
    Main window in View.
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.mainWindow = QMainWindow();
        self._createWidgets();
        self.mainWindow.setCentralWidget(self.mainWidget);
        self.mainWindow.show();
        
    def _createWidgets(self):
        '''
        '''
        self.mainWidget = QWidget(self.mainWindow)
        self.videoWidget = VideoWidget()
        self.layout = QGridLayout();
        self.layout.addWidget(self.videoWidget,0,0)
        self.mainWidget.setLayout(self.layout)
        