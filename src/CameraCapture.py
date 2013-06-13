'''
Created on Jun 12, 2013

@author: mcopik
'''
import cv2.cv as cv
import time
class ConversionType(set):
    AVERAGE = 1,
    LIGHTNESS = 2,
    LUMINOSITY = 3

def convertToGrayScaleAvg(img):
    mat = cv.CreateMat(img.width, img.height,cv.CV_32FC1)
    for i in range(img.width):
        for j in range(img.height):
            value = 0.0
            for k in range(3):
                value += img[i,j][k]
            mat[i,j] = value/3
    return mat

def convertToGrayScaleLightness(img):
    mat = cv.CreateMat(img.width, img.height,cv.CV_32FC1)
    for i in range(img.width):
        for j in range(img.height):
            min_value = min(img[i,j])
            max_value = max(img[i,j])
            mat[i,j] = (min_value+max_value)/2
    return mat

def convertToGrayScaleLuminosity(img):
    weights = [0.21,0.71,0.08]
    mat = cv.CreateMat(img.width, img.height,cv.CV_32FC1)
    for i in range(img.width):
        for j in range(img.height):
            value = 0.0
            for k in range(3):
                value += img[i,j][k]*weights[k]
            mat[i,j] = value
    return mat

def captureImg(size,filename,extension,number,start_number=0):
    cv.NamedWindow("camera", 1)
    capture = cv.CaptureFromCAM(0)
    start_time = time.clock()
    cur_time = 0
    counter = 0
    start = False
    while True:
        img = cv.QueryFrame(capture)
        cv.ShowImage("camera", img)
        if start == True:
            cur_time = time.clock() - start_time

        key = cv.WaitKey(10)
        if key == 27:
            break
        if key == 32:
            start = True
            start_time = time.clock()
        if counter == number:
            break
        if start and cur_time > 5*(counter+1):
            thumbnail = cv.CreateMat(size[0], size[1], cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            cv.SaveImage(filename+str(counter+start_number)+extension, thumbnail)
            counter += 1
            print 'Captured!'

def getImg(capture,size):
    #cv.NamedWindow("camera", 1)
    start_time = time.clock()
    cur_time = 0
    while True:
        cur_time = time.clock() - start_time
        key = cv.WaitKey(10)
        if cur_time > 4:
            break
    img = cv.QueryFrame(capture)
        #if start == True:
    #        cur_time = time.clock() - start_time

    #    key = cv.WaitKey(10)
    #   if key == 27:
    #        break
     #   if key == 32:
      #      start = True
       #     start_time = time.clock()
        #if start and cur_time > 4:
            
            #cv.ShowImage("camera", img)
    thumbnail = cv.CreateMat(size[0], size[1], cv.CV_8UC3)
    cv.Resize(img, thumbnail)
    gray = convertToGrayScaleLuminosity(thumbnail)
    data = []
    for i in range(size[0]):
        for j in range(size[1]):
            data.append(gray[i,j]/255)
    return data
        
def convertImg(src,dst,conversion):
    img = cv.LoadImage(src)
    thumbnail = cv.CreateMat(128, 128, cv.CV_8UC3)
    cv.Resize(img, thumbnail)
    if conversion == ConversionType.AVERAGE:
        cv.SaveImage(dst, convertToGrayScaleAvg(thumbnail))
    elif conversion == ConversionType.LIGHTNESS:
        cv.SaveImage(dst, convertToGrayScaleLightness(thumbnail))
    else:
        cv.SaveImage(dst, convertToGrayScaleLuminosity(thumbnail))

def loadImg(src,scale = [0,0]):
    img = cv.LoadImageM(src)
    if scale[0] != 0:
        thumbnail = cv.CreateMat(scale[0],scale[1],cv.CV_8UC3)
        cv.Resize(img, thumbnail)
        img = thumbnail
    gray = convertToGrayScaleLuminosity(img)
    for i in range(gray.width):
        for j in range(gray.height):
            gray[i,j] = gray[i,j]/255.0
    cv.SaveImage("spec"+src, gray)
    return gray