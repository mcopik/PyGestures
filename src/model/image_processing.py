'''
Created on Jul 1, 2013

@author: biomedyczna
'''

import cv2.cv   as  cv
import numpy    as  np

class ImageProcessing(object):
    '''
    classdocs
    '''
    #scale
    scale_width = 0
    scale_height = 0
    # ycbcr skin detection
    DEFAULT_CB_MIN = 80
    DEFAULT_CB_MAX = 135
    DEFAULT_CR_MIN = 130
    DEFAULT_CR_MAX = 180
    cb_min = DEFAULT_CB_MIN
    cb_max = DEFAULT_CB_MAX
    cr_min = DEFAULT_CR_MIN
    cr_max = DEFAULT_CR_MAX
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
    def yCbCrSkinDetection(self,func=None):
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            #constant algorithm data!
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
                    if not (val[1] > self.cb_min and val[1] < self.cb_max and
                            val[2] > self.cr_min and val[2] < self.cr_max):
                        img[j,i] = [0,0,0]
            return img
        return wrapper
       
    