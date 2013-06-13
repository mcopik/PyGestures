from PyQt4.QtGui import QApplication,QMainWindow,QWidget,QVBoxLayout,QPushButton,QLineEdit,QMessageBox
from PyQt4 import Qt
from pyqt import VideoWidget
from NeuralNetwork import NeuralNetwork
from CameraCapture import getImg
import sys

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
    def train(self):
        self.image_width = int(self.width_edit.text())
        self.image_height = int(self.height_edit.text())
        self.neural_network = NeuralNetwork([self.image_width,self.image_height],int(self.hidden_edit.text()),int(self.iterations_edit.text()),int(self.percent_edit.text()))
        self.capture_button.setEnabled(True)
    def capture(self):
        img = getImg(self.camera._capture,[self.image_width,self.image_height])
        QMessageBox.about(self.mainWindow,"Result!","Class: %s" % (self.neural_network.classify(img)))
        return
        
if __name__ == '__main__':
    main = Main(sys.argv)
#    widget = VideoWidget()
#    widget.setWindowTitle('PyQt - OpenCV Test')
#    widget.show()
#    hellobutton = QPushButton("Say",None)
#    hellobutton.show()
    
 
    sys.exit(main.exec_())