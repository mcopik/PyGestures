from PyQt4.QtGui import QApplication,QMainWindow,QWidget,QVBoxLayout,QPushButton
from pyqt import VideoWidget
import sys

class Main(QMainWindow):
    def __init__(self,parent = None):
        super(Main,self).__init__(parent)
        
        self.centralWidget = QWidget(self)
        self.vLayout = QVBoxLayout(self.centralWidget)
        self.widget = VideoWidget()
        self.button_add = QPushButton(self.centralWidget)
        self.button_add.setText('Button')
        self.vLayout.addWidget(self.widget)
        self.vLayout.addWidget(self.button_add)
        self.setCentralWidget(self.centralWidget)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
#    widget = VideoWidget()
#    widget.setWindowTitle('PyQt - OpenCV Test')
#    widget.show()
#    hellobutton = QPushButton("Say",None)
#    hellobutton.show()
    
 
    sys.exit(app.exec_())