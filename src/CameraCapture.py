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

def captureImg(size,filename,number,extension):
    cv.NamedWindow("camera", 1)
    capture = cv.CaptureFromCAM(0)
    start_time = time.clock()
    counter = 0
    while True:
        img = cv.QueryFrame(capture)
        cv.ShowImage("camera", img)
        start_time += time.clock() - start_time

        if cv.WaitKey(10) == 27:
            break
        if counter == number:
            break
        if start_time > 2*(counter+1):
            print start_time
            thumbnail = cv.CreateMat(size[0], size[1], cv.CV_8UC3)
            cv.Resize(img, thumbnail)
            cv.SaveImage(filename+str(counter)+extension, thumbnail)
            counter += 1
        
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

def loadImg(src):
    img = cv.LoadImageM(src)
    gray = convertToGrayScaleLuminosity(img)
    for i in range(gray.width):
        for j in range(gray.height):
            gray[i,j] = gray[i,j]/255.0
    cv.SaveImage("spec"+src, gray)
    return gray