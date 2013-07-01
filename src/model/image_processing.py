'''
Created on Jul 1, 2013

@author: biomedyczna
'''

import cv2.cv   as  cv
import cv2
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
    # blur
    DEFAULT_GAUSSIAN_SIZE = 3
    gaussian_size = DEFAULT_GAUSSIAN_SIZE
    DEFAULT_MEDIAN_SIZE = 5
    median_size = DEFAULT_MEDIAN_SIZE
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
        '''
        "Hand gesture recognition using a neural network shape fitting technique"
        '''
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
    def hsvSkinDetection(self,func = None):
        '''
        '''
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            hsv_image = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv.CvtColor(img, hsv_image, cv.CV_BGR2HSV)

            #COLOR_MIN = cv.Scalar(0,10,60)
            #COLOR_MAX = cv.Scalar(20,150,255)
            #ret = cv.CreateImage(cv.GetSize(hsv_image), 8, 1)
            #cv.InRangeS(hsv_image, COLOR_MIN, COLOR_MAX, ret)
            
            for i in range(img.width):
                for j in range(img.height):
                    pixel = hsv_image[j,i]
                    #print pixel
                    if not (pixel[0] > 0 and pixel[0] < 25 and
                            pixel[1] > 58 and pixel[1] < 174):
                        img[j,i] = [0,0,0]
            return img
            #return ret
        return wrapper    
    def gaussianBlur(self,func = None):
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            dst = cv.CreateImage(cv.GetSize(img),8,len(img[0,0]))
            cv.Smooth(img,dst,cv.CV_GAUSSIAN,self.gaussian_size,self.gaussian_size)
            return dst
        return wrapper
    def medianFilter(self,func = None):
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            dst = cv.CreateImage(cv.GetSize(img),8,len(img[0,0]))
            cv.Smooth(img,dst,cv.CV_MEDIAN,self.median_size,self.median_size)
            return dst
        return wrapper