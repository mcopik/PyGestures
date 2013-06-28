'''
Created on Jun 28, 2013

@author: mcopik
'''
from PyQt4.QtGui import QApplication

class MainApplication(QApplication):
    '''
    Main application of PyGestures
    '''


    def __init__(self,args):
        '''
        Constructor
        '''
        QApplication.__init__(self,args)
        