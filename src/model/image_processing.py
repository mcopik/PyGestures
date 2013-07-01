'''
Created on Jul 1, 2013

@author: biomedyczna
'''

import cv2.cv   as  cv

class ImageProcessing(object):
    '''
    classdocs
    '''

    scale_width = 0
    scale_height = 0
    def __init__(self):
        '''
        Constructor
        '''
        pass
    def scale(self,func = None):
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            thumbnail = cv.CreateMat(self.scale_width,self.scale_height,cv.CV_8UC3)
            cv.Resize(img,thumbnail)
            return thumbnail
        return wrapper
    def gray(self,func = None):
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            gray_image = cv.CreateImage(cv.GetSize(img), 8, 1)
            cv.CvtColor(img, gray_image, cv.CV_RGB2GRAY)
            return gray_image
        return wrapper
    