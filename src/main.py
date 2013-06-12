#from cv2 import *
import cv2.cv as cv
import time

cv.NamedWindow("camera", 1)

capture = cv.CaptureFromCAM(0)

while True:
    img = cv.QueryFrame(capture)
    cv.ShowImage("camera", img)
    if cv.WaitKey(10) == 27:
        cv.SaveImage("image.bmp", img)
        break
    
# initialize the camera
#frame = cv.CaptureFromCAM(0)
#img = cv.QueryFrame(frame)
#cam = VideoCapture(0)   # 0 -> index of camera
#s, img = cam.read()
    # frame captured without any errors
#cv.NamedWindow("cam-test",cv.CV_WINDOW_AUTOSIZE)
#cv.ShowImage("cam-test", img)
#cv.namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
#imshow("cam-test",img)
#cv.WaitKey(0)
#cv.DestroyWindow("cam-test")