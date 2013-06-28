from PyQt4.QtGui import QApplication,QMainWindow,QWidget,QVBoxLayout,QPushButton,QLineEdit,QMessageBox
from PyQt4 import Qt
from VideoWidget import VideoWidget
from NeuralNetwork import NeuralNetwork
from CameraCapture import getImg,getRealImg
from math import sqrt
from numpy import convolve
from scipy.ndimage.filters import convolve1d
import cv2.cv as cv
import numpy as np
import cv2
import sys
#from numpy.distutils.command.config import GrabStdout
#from matplotlib.pyplot import gray

class Main(QApplication):
    def __init__(self,args):
        QApplication.__init__(self,args)
        self.mainWindow = QMainWindow()
        self.centralWidget = QWidget(self.mainWindow)
        self.vLayout = QVBoxLayout(self.centralWidget)
        self.createWidgets()
        self.mainWindow.setCentralWidget(self.centralWidget)
        self.mainWindow.show()
    def createWidgets(self):
        self.camera = VideoWidget()
        self.iterations_edit = QLineEdit("20")
        self.width_edit = QLineEdit("40")
        self.height_edit = QLineEdit("40")
        self.hidden_edit = QLineEdit("1000")
        self.percent_edit = QLineEdit("25")
        self.sobel_button = QPushButton(self.centralWidget)
        self.connect(self.sobel_button, Qt.SIGNAL("clicked()"),self.filter)
        self.button_train = QPushButton(self.centralWidget)
        self.button_train.setText('Train network')
        self.button_train.clicked.connect(self.train)
        self.capture_button = QPushButton(self.centralWidget)
        self.capture_button.setText('Capture(after 4 sec)')
        self.capture_button.setEnabled(False)
        self.connect(self.capture_button,Qt.SIGNAL("clicked()"),self.capture)
        self.vLayout.addWidget(self.camera)
        self.vLayout.addWidget(self.iterations_edit)
        self.vLayout.addWidget(self.width_edit)
        self.vLayout.addWidget(self.height_edit)
        self.vLayout.addWidget(self.hidden_edit)
        self.vLayout.addWidget(self.percent_edit)
        self.vLayout.addWidget(self.button_train)
        self.vLayout.addWidget(self.capture_button)
        self.vLayout.addWidget(self.sobel_button)
    def train(self):
        self.image_width = int(self.width_edit.text())
        self.image_height = int(self.height_edit.text())
        self.neural_network = NeuralNetwork([self.image_width,self.image_height],int(self.hidden_edit.text()),int(self.iterations_edit.text()),int(self.percent_edit.text()))
        self.capture_button.setEnabled(True)
    def capture(self):
        img = getImg(self.camera._capture,[self.image_width,self.image_height])
        out = self.neural_network.classify(img)
        QMessageBox.about(self.mainWindow,"Result!","Class: %s" % (out))
        out = out.argmax(axis=1)
        QMessageBox.about(self.mainWindow,"Classification result","Class : %s" % (self.neural_network.classes[out]))
        return
    def filter(self):
        img = getRealImg(self.camera._capture)
        #img = cv.LoadImageM("testdata4.bmp")
        gray_image = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.CvtColor(img, gray_image, cv.CV_RGB2GRAY)
        ar = np.asarray(img)
        vec = np.array([16,128,128])
        mat = np.array([[65.481,128.553,24.966],
                        [-37.797,-74.203,112],
                        [112,-93.786,-18.214]])

        for i in range(img.width):
            for j in range(img.height):
                rgb = np.array([img[j,i][2],img[j,i][1],img[j,i][0]])
                #print i,j,rgb
                val = mat.dot(rgb)
                val = val/256 + vec
                if not (val[1] > 80 and val[1] < 135 and
                    val[2] > 130 and val[2] < 180):
                    #print val[0],val[1],val[2]
                    gray_image[j,i] = 0
        
        #cv.Smooth(gray_image,gray_image,cv.CV_GAUSSIAN,5,5)            
        cv.SaveImage("test2.png", gray_image)
        cv.NamedWindow("Captured image")
        cv.ShowImage("Captured image", gray_image)
    def sobel(self):
        img = getRealImg(self.camera._capture)
        #img = cv.LoadImageM("testdata60.bmp")
        
        gray_image = cv.CreateImage(cv.GetSize(img), 8, 1)
        cv.CvtColor(img, gray_image, cv.CV_RGB2GRAY)
        #for i in range(gray_image.width):
        #    for j in range(gray_image.height):
        #        gray_image[j,i] = 255-gray_image[j,i]
        #gray_image = img
        df_dx = cv.CreateImage(cv.GetSize(img),8,1)
        #cv.Sobel(gray_image, df_dx, 1, 0)
        ar = np.asarray(gray_image)
        #gray_image = cv2.blur(ar,3)
        #cv.Smooth(gray_image,df_dx,cv.CV_GAUSSIAN,5,5)
        df_dx = gray_image
        #cv.Canny(df_dx,gray_image,50,150,3)
        mat = cv.GetMat(df_dx)
        mat = convolve1d(mat,[0,-1,1],axis=-1,mode='constant')
        #size = cv.GetSize(gray_image)
        #print size
        #print cv.GetSize(df_dx)
        A = [
     [1  ,   2,     3],
     [3 ,    4 ,    5],
     [1,     2  ,   3]]
        A = convolve1d(A,[0,-1,1],axis=-1,mode='constant')
        print A[0][1]," ",A[0][2]
        first = False
        for i in range(gray_image.height):
            for j in range(gray_image.width):
                gray_image[i,j] = mat[i,j]
        cv.NamedWindow("Captured image")
        cv.ShowImage("Captured image", gray_image)
"""                if gray_image[i,j] > 120:
                    if first == False:
                        first = True
                    else:
                        first = False
                else:
                    if first == True:
                        gray_image[i,j] = 255
            first = False"""
                #print i,j
                #if i == 0:
                #    df_dx[j,i] = -gray_image[j,i] + gray_image[j,i+1]
                #       if i == gray_image.width -1 or j == gray_image.height -1:
                #    df_dx[j,i] = abs(-gray_image[j,i]) + abs(gray_image[j,i])
                
                #      else:
                #         dx_fx[j,i]
                #    df_dx[j,i] = abs(-gray_image[j,i] + gray_image[j,i+1])+abs(gray_image[j,i]-gray_image[j+1,i])
        
#if __name__ == '__main__':
#    main = Main(sys.argv)
#    widget = VideoWidget()
#    widget.setWindowTitle('PyQt - OpenCV Test')
#    widget.show()
#    hellobutton = QPushButton("Say",None)
#    hellobutton.show()
    
 
#    sys.exit(main.exec_())