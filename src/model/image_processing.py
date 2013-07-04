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
            if func is not None:
                image = func(image,param)
            img = cv.CreateImage(cv.GetSize(image),8,len(image[0,0]))
            cv.Copy(image, img)
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
            img = cv.CreateImage(cv.GetSize(image),8,len(image[0,0]))
            cv.Copy(image, img)
            if func is not None:
                img = func(image,param)
            hsv_image = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv.CvtColor(img, hsv_image, cv.CV_BGR2HSV)
            #COLOR_MIN = cv.Scalar(0,10,60)
            #COLOR_MAX = cv.Scalar(25,173,229)
            #ret = cv.CreateImage(cv.GetSize(hsv_image), 8, 1)
            #cv.InRangeS(hsv_image, COLOR_MIN, COLOR_MAX, ret)
            
            for i in range(img.width):
                for j in range(img.height):
                    pixel = hsv_image[j,i]
                    #print pixel
                    if not (pixel[0] > 0 or pixel[0] < 25 and
                            pixel[1] > 58 and pixel[1] < 174):
                        img[j,i] = [0,0,0]
            return img
            #return ret
        return wrapper    
    def combinedSkinDetection(self,func = None):
        '''
        '''
        def wrapper(image,param = None):
            img = cv.CreateImage(cv.GetSize(image),8,len(image[0,0]))
            cv.Copy(image, img)
            if func is not None:
                img = func(image,param)
            hsv_image = cv.CreateImage(cv.GetSize(img), 8, 3)
            #cv.Convert(img, hsv_image,cv.CV_32FC3)
            ar = np.asarray(img[:,:])
            hsv_image = np.float32(ar)
            hsv_image = cv.fromarray(hsv_image)
            cv.CvtColor(hsv_image, hsv_image, cv.CV_BGR2HSV)
            hsv_image = np.array(hsv_image)
            cv2.normalize(hsv_image, hsv_image,0,255,cv2.NORM_MINMAX,cv.CV_32FC3)
            ycbcr_image = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv.CvtColor(img, ycbcr_image, cv.CV_BGR2YCrCb)
            #COLOR_MIN = cv.Scalar(0,10,60)
            #COLOR_MAX = cv.Scalar(20,150,255)
            #ret = cv.CreateImage(cv.GetSize(hsv_image), 8, 1)
            #cv.InRangeS(hsv_image, COLOR_MIN, COLOR_MAX, ret)
            
            for i in range(img.width):
                for j in range(img.height):
                    bgr_pixel = img[j,i]
                    hsv_pixel = hsv_image[j,i]
                    ycbcr_pixel = ycbcr_image[j,i]
                    if not (hsv_pixel[0] < 25 or hsv_pixel[0] > 230):
                        img[j,i] = [0,0,0]
                    '''
                    if not (bgr_pixel[2] > 95 and bgr_pixel[1] > 40 and bgr_pixel[0] > 20
                        and (max(bgr_pixel) - min(bgr_pixel)) > 15 
                        and abs(bgr_pixel[2] - bgr_pixel[1]) > 15 and bgr_pixel[2] > bgr_pixel[1]
                        and bgr_pixel[2] > bgr_pixel[0]):
                        if not (bgr_pixel[2] > 220 and bgr_pixel[1] > 210 and bgr_pixel[0] > 170
                                and abs(bgr_pixel[2] - bgr_pixel[1]) <= 15 and bgr_pixel[2] > bgr_pixel[0]
                                and bgr_pixel[1] > bgr_pixel[0]):
                            img[j,i] = [0,0,0];
                    '''
                    Cr = ycbcr_pixel[1]
                    Cb = ycbcr_pixel[2]
                    e3 = Cr <= 1.5862*Cb+20;
                    e4 = Cr >= 0.3448*Cb+76.2069;
                    e5 = Cr >= -4.5652*Cb+234.5652;
                    e6 = Cr <= -1.15*Cb+301.75;
                    e7 = Cr <= -2.2857*Cb+432.85;
                    if not (e3 and e4 and e5 and e6 and e7):
                        img[j,i] = [0,0,0]
                    #RGB
                    #print hsv_pixel
                    #if not (( hsv_pixel[0] < 25) and
                    #        (hsv_pixel[1] > 58 and hsv_pixel[1] < 174)):
                    #    img[j,i] = [0,0,0]
            return img
            #return ret
        return wrapper
    def processContour(self,func = None):
        '''
        problem with returned value!
        '''
        def wrapper(image,param = None):
            if func is not None:
                image = func(image,param)
            img = cv.CreateImage(cv.GetSize(image),8,1)
            cv.CvtColor(image,img, cv.CV_BGR2GRAY)
            for i in range(img.width):
                for j in range(img.height):
                    if img[j,i] != 0:
                        img[j,i] = 255
            ar = np.asarray(img[:,:])
            contours, hierarchy = cv2.findContours(ar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            index = 0
            size = 0
            second_index = 0
            new_image = np.zeros((img.height,img.width),np.uint8)
            for i in range(len(contours)):
                if len(contours[i]) > size:
                    size = len(contours[i])
                    
                    second_index = index
                    index = i
            print index,size
            cv2.drawContours(new_image, contours[index], -1, cv.Scalar(255), 1)
            cv2.drawContours(new_image, contours, index, cv.Scalar(255), -1)
            cv_image = cv.CreateImage(cv.GetSize(img),8,1)
            print cv.GetSize(img)
            print cv.GetSize(cv_image)
            for i in range(img.width):
                for j in range(img.height):
                    cv_image[j,i] = new_image[j,i]
            #cv2.drawContours(new_image, contours, second_index, cv.Scalar(255), 1)
            #cv2.drawContours(new_image, contours, second_index, cv.Scalar(255), -1)
            #i = 0
            #print contours
            #while contours:
            #    if len(contours) > size:
            #        index = i
            #        size = len(contours)
            #    i += 1
            #    contours = contours.h_next()
            return cv_image
        return wrapper
    def ellipseSkinDetection(self,func = None):
        '''
        doesn't work! why not? i have no idea
        '''
        def wrapper(image,param = None):
            img = image
            if func is not None:
                img = func(image,param)
            skinMat = cv.CreateMat(256, 256, cv.CV_8UC1)
            cv.Ellipse(skinMat, (113,156), (23,15), 43.0, 0, 360.0, cv.Scalar(255,255,255),-1)
            ycbcr_image = cv.CreateImage(cv.GetSize(img), 8, 3)
            cv.CvtColor(img, ycbcr_image, cv.CV_BGR2YCrCb)
            #COLOR_MIN = cv.Scalar(0,10,60)
            #COLOR_MAX = cv.Scalar(20,150,255)
            #ret = cv.CreateImage(cv.GetSize(hsv_image), 8, 1)
            #cv.InRangeS(hsv_image, COLOR_MIN, COLOR_MAX, ret)
            
            for i in range(img.width):
                for j in range(img.height):
                    ycbcr_pixel = ycbcr_image[j,i]
                    if not skinMat[ycbcr_pixel[1],ycbcr_pixel[2]] > 0:
                        img[j,i] = [0,0,0]
                    #RGB
                    #if not (pixel[0] > 0 or pixel[0] < 25 and
                    #        pixel[1] > 58 and pixel[1] < 174):
                    #    img[j,i] = [0,0,0]
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
            if func is not None:
                image = func(image,param)
            img = image
            dst = cv.CreateImage(cv.GetSize(img),8,len(img[0,0]))
            cv.Smooth(img,dst,cv.CV_MEDIAN,self.median_size,self.median_size)
            return dst
        return wrapper
    def medianFilter2(self,func = None):
        def wrapper(image,param = None):
            if func is not None:
                image = func(image,param)
            img = image
            return cv2.medianBlur(np.asarray(img, np.int8),self.median_size)
        return wrapper