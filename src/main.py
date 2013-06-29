'''
Created on Jun 28, 2013

@author: mcopik
'''
from sys import argv
from sys import exit

from controller.main_application import MainApplication

if __name__ == '__main__':
    app = MainApplication(argv)
    exit(app.exec_())